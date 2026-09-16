package com.splat.mobile3dgs.hardware

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.util.Log

enum class HardwareTier(val tierName: String, val description: String) {
    TIER_1_FLAGSHIP("Tier 1 (Flagship)", "Snapdragon 8 Gen 2 / 12GB+ RAM: Full On-Device 3DGS (3k steps, 720p)"),
    TIER_2_BALANCED("Tier 2 (Balanced)", "8GB - 11GB RAM: Optimized On-Device 3DGS (3.5k steps, 720p)"),
    TIER_3_STANDALONE("Tier 3 (Standalone)", "<8GB RAM: Standalone Direct Photometric Splatting & 360p Fast Refine")
}

data class TrainingProfile(
    val tier: HardwareTier,
    val totalSteps: Int,
    val maxResolution: Int,
    val maxGaussians: Int,
    val refineEvery: Int,
    val exportEvery: Int,
    val useHalfPrecision: Boolean,
    val totalRamGb: Float,
    val socModel: String,
    val supportsDirectPhotometricSplat: Boolean = true
)

/**
 * How hard the training thread should be duty-cycled right now.
 *
 * [throttleMs] is slept inside the native progress callback, which fires roughly
 * every 5 training steps; [paused] holds the thread until conditions improve.
 */
data class DutyCycle(
    val throttleMs: Int,
    val paused: Boolean,
    val reason: String
) {
    val isFullSpeed: Boolean get() = throttleMs == 0 && !paused
}

/** Whether a run may start at all. */
data class StartVerdict(val allowed: Boolean, val reason: String)

object DeviceCapabilityManager {
    private const val TAG = "DeviceCapabilityManager"

    /** Below this (and not charging) a run is refused outright. */
    const val MIN_START_BATTERY_PCT = 20

    /** Below this (and not charging) an in-flight run is held. */
    const val PAUSE_BATTERY_PCT = 15

    fun getDeviceProfile(context: Context): TrainingProfile {
        val totalRamGb = getTotalRamGb(context)
        val soc = getSocModel()

        val socLower = soc.lowercase()
        val isBudgetSoc = socLower.contains("sm4") || socLower.contains("sm6") ||
                socLower.contains("4450") || socLower.contains("6375") ||
                socLower.contains("helio") || socLower.contains("g99") ||
                socLower.contains("dimensity 6") || socLower.contains("dimensity 7")

        val tier = when {
            isBudgetSoc -> HardwareTier.TIER_3_STANDALONE
            totalRamGb >= 10.5f -> HardwareTier.TIER_1_FLAGSHIP
            totalRamGb >= 6.8f -> HardwareTier.TIER_2_BALANCED
            else -> HardwareTier.TIER_3_STANDALONE
        }

        Log.i(TAG, "Device profile evaluated: SoC=$soc (budget=$isBudgetSoc), RAM=${"%.1f".format(totalRamGb)} GB -> $tier")

        return when (tier) {
            HardwareTier.TIER_1_FLAGSHIP -> TrainingProfile(
                tier = tier,
                // 1080p costs 2.25x the pixels of 720p per step and was the main
                // reason runs took 30+ minutes and thermally throttled. 720p with
                // good parallax beats 1080p with poor parallax; 7000 steps and
                // 1080p remain available as an explicit choice in the dialog.
                // 3000 steps was far too few. Reference 3DGS trains 30k and treats
                // 7k as the first acceptable checkpoint; a measured run produced
                // 493k Gaussians in 3000 steps, so everything densified after ~step
                // 2000 was still an unoptimised blob. That reads as fog.
                totalSteps = 7000,
                maxResolution = 720,
                maxGaussians = 1000000,
                // Densification cadence is the only growth lever brush-c exposes
                // (TrainOptions has no max_splats). Stretching it trades raw splat
                // COUNT for splat QUALITY: fewer Gaussians each get more gradient
                // steps, and every step is cheaper because there is less to
                // rasterise -- sharper AND faster, rather than a pure time cost.
                refineEvery = 250,
                exportEvery = 1000,
                useHalfPrecision = true,
                totalRamGb = totalRamGb,
                socModel = soc,
                supportsDirectPhotometricSplat = true
            )
            HardwareTier.TIER_2_BALANCED -> TrainingProfile(
                tier = tier,
                totalSteps = 3500,
                maxResolution = 720,
                maxGaussians = 400000,
                refineEvery = 100,
                exportEvery = 500,
                useHalfPrecision = true,
                totalRamGb = totalRamGb,
                socModel = soc,
                supportsDirectPhotometricSplat = true
            )
            HardwareTier.TIER_3_STANDALONE -> TrainingProfile(
                tier = tier,
                // Budget SoCs (e.g. Snapdragon 4/6 series, Mali-G52) need lightweight
                // parameters to ensure training completes in under ~45s without thermal
                // throttling or exceeding available user-space RAM.
                // 300 steps at 360p cannot reconstruct anything -- it just produced a
                // few hundred splats. Budget SoCs are slow, not incapable.
                totalSteps = 1500,
                maxResolution = 540,
                maxGaussians = 80000,
                refineEvery = 100,
                exportEvery = 300,
                useHalfPrecision = false,
                totalRamGb = totalRamGb,
                socModel = soc,
                supportsDirectPhotometricSplat = true
            )
        }
    }

    /**
     * Refinement cadence to actually hand the engine for a given budget.
     *
     * The engine's C ABI has no Gaussian cap (`TrainConfig::max_splats` is not
     * reachable through `TrainOptions`, has no env binding, and the `args.txt`
     * config `brush-process` would otherwise honour is discarded by `brush-c`).
     * Densification therefore only happens on refinement ticks, so the cadence
     * is the one growth lever we have: stretching it cuts the number of
     * densification events, which is what produced a 578k-splat overfit on a
     * device budgeted for 80k.
     *
     * This is a damping heuristic, not a hard cap -- the hard cap is applied by
     * decimating the finished model before it is published.
     */
    fun refineEveryFor(profile: TrainingProfile): Int {
        val base = profile.refineEvery.coerceAtLeast(50)
        return when (profile.tier) {
            HardwareTier.TIER_1_FLAGSHIP -> base
            HardwareTier.TIER_2_BALANCED -> (base * 1.5f).toInt()
            HardwareTier.TIER_3_STANDALONE -> base * 3
        }.coerceIn(50, 2000)
    }

    /**
     * Checkpoint cadence: frequent enough that a killed run loses little, rare
     * enough that PLY serialisation does not dominate the step time.
     */
    fun exportEveryFor(profile: TrainingProfile, steps: Int): Int {
        val target = maxOf(500, steps / 5)
        // Never checkpoint *less* often than the device profile asks for.
        val hinted = if (profile.exportEvery > 0) minOf(target, maxOf(profile.exportEvery, 100)) else target
        return hinted.coerceIn(1, maxOf(1, steps))
    }

    fun getTotalRamGb(context: Context): Float {
        val actManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        actManager?.getMemoryInfo(memInfo)
        return (memInfo.totalMem / (1024.0 * 1024.0 * 1024.0)).toFloat()
    }

    fun getAvailableRamGb(context: Context): Float {
        val actManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        actManager?.getMemoryInfo(memInfo)
        return (memInfo.availMem / (1024.0 * 1024.0 * 1024.0)).toFloat()
    }

    fun getBatteryLevel(context: Context): Int {
        val filter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        val batteryStatus: Intent? = context.registerReceiver(null, filter)
        val level = batteryStatus?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = batteryStatus?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
        return if (level >= 0 && scale > 0) (level * 100) / scale else 100
    }

    fun isCharging(context: Context): Boolean {
        return try {
            val filter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
            val status = context.registerReceiver(null, filter)
                ?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
            status == BatteryManager.BATTERY_STATUS_CHARGING ||
                status == BatteryManager.BATTERY_STATUS_FULL
        } catch (e: Exception) {
            Log.w(TAG, "Could not read charging state: ${e.message}")
            false
        }
    }

    /**
     * Raw `PowerManager` thermal status (0 NONE .. 6 SHUTDOWN), or 0 on API < 29
     * where the signal does not exist.
     */
    fun getThermalStatus(context: Context): Int {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val pm = context.getSystemService(Context.POWER_SERVICE) as? PowerManager
            return pm?.currentThermalStatus ?: PowerManager.THERMAL_STATUS_NONE
        }
        return PowerManager.THERMAL_STATUS_NONE
    }

    fun isThermalThrottling(context: Context): Boolean =
        getThermalStatus(context) >= PowerManager.THERMAL_STATUS_SEVERE

    /**
     * Duty cycle for the current thermal + battery state.
     *
     * Sustained training halves throughput purely from heat (measured ~3.0 ->
     * ~1.4 steps/s), and past that the SoC throttles everything including the
     * UI. Backing off early keeps the average higher than being throttled by
     * the kernel, and keeps the phone usable.
     */
    fun currentDutyCycle(context: Context): DutyCycle {
        val battery = getBatteryLevel(context)
        val charging = isCharging(context)

        if (battery < PAUSE_BATTERY_PCT && !charging) {
            return DutyCycle(0, true, "Paused: battery $battery% (resumes above $PAUSE_BATTERY_PCT% or on charge)")
        }

        return when (getThermalStatus(context)) {
            // NONE / LIGHT: the device is coping.
            0, 1 -> DutyCycle(0, false, "Running at full speed")
            // MODERATE: shed a little load before the kernel does it for us.
            2 -> DutyCycle(250, false, "Easing off: device is warm")
            // SEVERE: throughput is already collapsing; back off hard.
            3 -> DutyCycle(1200, false, "Slowed down: device is hot")
            // CRITICAL and worse: stop entirely until it cools.
            else -> DutyCycle(0, true, "Paused: device is too hot, waiting to cool")
        }
    }

    /** Whether a new run may start right now. */
    fun canStartTraining(context: Context): StartVerdict {
        val battery = getBatteryLevel(context)
        val charging = isCharging(context)
        if (battery < MIN_START_BATTERY_PCT && !charging) {
            return StartVerdict(
                false,
                "Battery is $battery%. Training needs at least $MIN_START_BATTERY_PCT% " +
                    "or a charger -- a full run can take tens of minutes at high load."
            )
        }
        val thermal = getThermalStatus(context)
        if (thermal >= 4) {
            return StartVerdict(
                false,
                "The device is too hot to start training (thermal status $thermal). " +
                    "Let it cool down and try again."
            )
        }
        return StartVerdict(true, "")
    }

    private fun getSocModel(): String {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            Build.SOC_MODEL.ifEmpty { Build.HARDWARE }
        } else {
            Build.HARDWARE
        }
    }
}
