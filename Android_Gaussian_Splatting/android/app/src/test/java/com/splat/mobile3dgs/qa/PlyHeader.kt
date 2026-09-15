package com.splat.mobile3dgs.qa

import java.io.File

/**
 * Minimal binary-PLY header parser, written to mirror the C++ parser in
 * `app/src/main/cpp/brush_bridge.cpp::convert_ply_to_splat` exactly, including its
 * type-size table. Its purpose is to catch the class of bug where the ASCII header
 * declares one set of properties and the binary body is written with a different
 * stride -- a silent corruption that turns a scan into noise rather than a crash.
 */
object PlyHeader {

    /** Same table as `convert_ply_to_splat`'s property-size switch. */
    fun sizeOf(type: String): Int = when (type) {
        "float", "float32", "int", "uint", "int32", "uint32" -> 4
        "double", "float64" -> 8
        "uchar", "uint8", "char", "int8" -> 1
        "short", "uint16", "int16" -> 2
        else -> 4
    }

    data class Property(val type: String, val name: String) {
        val sizeBytes: Int get() = sizeOf(type)
    }

    data class Parsed(
        val format: String,
        val vertexCount: Int,
        val properties: List<Property>,
        /** Byte length of the ASCII header, including the trailing "end_header\n". */
        val headerBytes: Int
    ) {
        val bytesPerVertex: Int get() = properties.sumOf { it.sizeBytes }
        val expectedFileBytes: Long get() = headerBytes.toLong() + vertexCount.toLong() * bytesPerVertex
        val propertyNames: List<String> get() = properties.map { it.name }

        /** Byte offset of [name] within one vertex record, or -1. */
        fun offsetOf(name: String): Int {
            var off = 0
            for (p in properties) {
                if (p.name == name) return off
                off += p.sizeBytes
            }
            return -1
        }
    }

    fun parse(bytes: ByteArray): Parsed {
        var format = ""
        var vertexCount = -1
        val props = mutableListOf<Property>()
        var i = 0
        var lineStart = 0
        var headerBytes = -1
        var sawPly = false

        while (i < bytes.size) {
            if (bytes[i] == '\n'.code.toByte()) {
                var end = i
                if (end > lineStart && bytes[end - 1] == '\r'.code.toByte()) end--
                val line = String(bytes, lineStart, end - lineStart, Charsets.US_ASCII).trim()
                val tok = line.split(Regex("\\s+")).filter { it.isNotEmpty() }
                when {
                    tok.isEmpty() -> {}
                    tok[0] == "ply" -> sawPly = true
                    tok[0] == "format" -> format = tok.drop(1).joinToString(" ")
                    tok[0] == "element" && tok.size >= 3 && tok[1] == "vertex" ->
                        vertexCount = tok[2].toIntOrNull() ?: -1
                    tok[0] == "property" && tok.size >= 3 -> props += Property(tok[1], tok[2])
                    tok[0] == "end_header" -> { headerBytes = i + 1 }
                }
                if (headerBytes >= 0) break
                lineStart = i + 1
            }
            i++
        }
        require(sawPly) { "not a PLY file: missing 'ply' magic line" }
        require(headerBytes >= 0) { "PLY header has no end_header line" }
        return Parsed(format, vertexCount, props, headerBytes)
    }

    fun parse(file: File): Parsed = parse(file.readBytes())
}
