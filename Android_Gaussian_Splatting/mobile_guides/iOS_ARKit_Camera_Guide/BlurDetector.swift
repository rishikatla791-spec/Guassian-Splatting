//
//  BlurDetector.swift
//  3DGS Camera Guide (iOS ARKit)
//
//  Computes Laplacian variance on CVPixelBuffer camera frames
//  using Apple's Accelerate / vImage framework to reject motion-blurred captures.
//

import Foundation
import AVFoundation
import Accelerate
import CoreImage

public class BlurDetector {
    
    /// Computes Laplacian variance blur score for a CVPixelBuffer frame.
    /// Higher variance (> 100) indicates a sharp image; low variance (< 100) indicates motion blur.
    public static func computeLaplacianVariance(pixelBuffer: CVPixelBuffer) -> Float {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let srcAddr = CVPixelBufferGetBaseAddress(pixelBuffer) else { return 0.0 }
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        
        var srcBuffer = vImage_Buffer(
            data: srcAddr,
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: bytesPerRow
        )
        
        // Allocate grayscale 8-bit buffer
        guard let grayData = malloc(height * width) else { return 0.0 }
        defer { free(grayData) }
        
        var grayBuffer = vImage_Buffer(
            data: grayData,
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: width
        )
        
        // Convert to Grayscale if BGRA
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        if pixelFormat == kCVPixelFormatType_32BGRA {
            var bgraToGrayMatrix: [int16] = [299, 587, 114, 0] // Rec. 601 coefficients
            vImageMatrixMultiply_ARGB8888ToPlanar8(&srcBuffer, &grayBuffer, &bgraToGrayMatrix, 1000, nil, 0, vImage_Flags(kvImageNoFlags))
        } else {
            // Assume 8-bit planar luminance (Y channel)
            memcpy(grayData, srcAddr, height * width)
        }
        
        // Apply 3x3 Laplacian Filter Kernel
        //  0   1   0
        //  1  -4   1
        //  0   1   0
        let laplacianKernel: [int16] = [
            0,  1,  0,
            1, -4,  1,
            0,  1,  0
        ]
        
        guard let lapData = malloc(height * width * MemoryLayout<Float>.size) else { return 0.0 }
        defer { free(lapData) }
        
        var lapBuffer = vImage_Buffer(
            data: lapData,
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: width * MemoryLayout<Float>.size
        )
        
        // Convolve and compute variance
        vImageConvolve_Planar8(&grayBuffer, &grayBuffer, nil, 0, 0, laplacianKernel, 3, 3, 1, 0, vImage_Flags(kvImageEdgeExtend))
        
        // Convert Planar8 to Float
        var zero: Float = 0.0
        var one: Float = 255.0
        vImageConvert_Planar8ToPlanarF(&grayBuffer, &lapBuffer, &zero, &one, vImage_Flags(kvImageNoFlags))
        
        // Compute Mean and Standard Deviation using vDSP
        let count = vDSP_Length(width * height)
        let floatPtr = lapData.assumingMemoryBound(to: Float.self)
        
        var mean: Float = 0.0
        var stdDev: Float = 0.0
        vDSP_normalize(floatPtr, 1, nil, 1, &mean, &stdDev, count)
        
        let variance = stdDev * stdDev
        return variance
    }
}
