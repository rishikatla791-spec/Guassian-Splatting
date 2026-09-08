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
    TIER_1_FLAGSHIP("Tier 1 (Flagship)", "Snapdragon 8 Gen 2 / 12GB+ RAM: Full On-Device 3DGS (7k steps, 1080p)"),
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

object DeviceCapabilityManager {
    private const val TAG = "DeviceCapabilityManager"

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
                totalSteps = 7000,
                maxResolution = 1080,
                maxGaussians = 1000000,
                refineEvery = 100,
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
                totalSteps = 300,
                maxResolution = 360,
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

    fun isThermalThrottling(context: Context): Boolean {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val powerManager = context.getSystemService(Context.POWER_SERVICE) as? PowerManager
            val thermalStatus = powerManager?.currentThermalStatus ?: PowerManager.THERMAL_STATUS_NONE
            return thermalStatus >= PowerManager.THERMAL_STATUS_SEVERE
        }
        return false
    }

    private fun getSocModel(): String {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            Build.SOC_MODEL.ifEmpty { Build.HARDWARE }
        } else {
            Build.HARDWARE
        }
    }
}
