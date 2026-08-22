//
//  DatasetExporter.swift
//  3DGS Camera Guide (iOS ARKit)
//
//  Exports captured photos alongside 4x4 camera transform matrices
//  and camera intrinsics to transforms.json format for direct 3DGS training.
//

import Foundation
import ARKit
import UIKit

public struct FrameMetaData: Codable {
    public let filePath: String
    public let transformMatrix: [[Float]] // 4x4 transform matrix
    public let sharpnessScore: Float
    public let timestamp: Double
}

public struct TransformsJSON: Codable {
    public let cameraAngleX: Float
    public let cameraAngleY: Float
    public let flX: Float
    public let flY: Float
    public let cX: Float
    public let cY: Float
    public let w: Int
    public let h: Int
    public let frames: [FrameMetaData]
    
    enum CodingKeys: String, CodingKey {
        case cameraAngleX = "camera_angle_x"
        case cameraAngleY = "camera_angle_y"
        case flX = "fl_x"
        case flY = "fl_y"
        case cX = "cx"
        case cY = "cy"
        case w, h
        case frames
    }
}

public class DatasetExporter {
    public private(set) var capturedFrames: [FrameMetaData] = []
    public let outputDirectory: URL
    
    public init(sessionName: String = "3dgs_session_\(Int(Date().timeIntervalSince1970))") {
        let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        self.outputDirectory = docsDir.appendingPathComponent(sessionName)
        
        try? FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true, attributes: nil)
        try? FileManager.default.createDirectory(at: outputDirectory.appendingPathComponent("images"), withIntermediateDirectories: true, attributes: nil)
    }
    
    /// Save a frame image and metadata
    public func saveCapturedFrame(image: UIImage, camera: ARCamera, sharpnessScore: Float) {
        let frameId = capturedFrames.count
        let imageName = String(format: "frame_%04d.jpg", frameId)
        let imageURL = outputDirectory.appendingPathComponent("images/\(imageName)")
        
        if let jpegData = image.jpegData(compressionQuality: 0.95) {
            try? jpegData.write(to: imageURL)
        }
        
        // Convert 4x4 transform matrix (Column-Major SIMD float4x4 -> 2D Array)
        let t = camera.transform
        let matrixArray: [[Float]] = [
            [t.columns.0.x, t.columns.1.x, t.columns.2.x, t.columns.3.x],
            [t.columns.0.y, t.columns.1.y, t.columns.2.y, t.columns.3.y],
            [t.columns.0.z, t.columns.1.z, t.columns.2.z, t.columns.3.z],
            [t.columns.0.w, t.columns.1.w, t.columns.2.w, t.columns.3.w]
        ]
        
        let metadata = FrameMetaData(
            filePath: "images/\(imageName)",
            transformMatrix: matrixArray,
            sharpnessScore: sharpnessScore,
            timestamp: Date().timeIntervalSince1970
        )
        
        capturedFrames.append(metadata)
    }
    
    /// Export final transforms.json file
    public func exportTransformsJSON(intrinsics: camera_matrix_t, imageSize: CGSize) -> URL? {
        let fx = intrinsics.columns.0.x
        let fy = intrinsics.columns.1.y
        let cx = intrinsics.columns.2.x
        let cy = intrinsics.columns.2.y
        
        let fovX = 2.0 * atan(Float(imageSize.width) / (2.0 * fx))
        let fovY = 2.0 * atan(Float(imageSize.height) / (2.0 * fy))
        
        let transforms = TransformsJSON(
            cameraAngleX: fovX,
            cameraAngleY: fovY,
            flX: fx,
            flY: fy,
            cX: cx,
            cY: cy,
            w: Int(imageSize.width),
            h: Int(imageSize.height),
            frames: capturedFrames
        )
        
        let jsonEncoder = JSONEncoder()
        jsonEncoder.outputFormatting = .prettyPrinted
        
        guard let jsonData = try? jsonEncoder.encode(transforms) else { return nil }
        let jsonURL = outputDirectory.appendingPathComponent("transforms.json")
        try? jsonData.write(to: jsonURL)
        
        return jsonURL
    }
}
