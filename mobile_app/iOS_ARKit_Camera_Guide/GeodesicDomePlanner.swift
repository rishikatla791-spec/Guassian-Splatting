//
//  GeodesicDomePlanner.swift
//  3DGS Camera Guide (iOS ARKit)
//
//  Computes 3D target nodes on a geodesic hemisphere around an object
//  and evaluates 6-DoF camera pose alignment in real time.
//

import Foundation
import ARKit
import SceneKit

public struct TargetNode {
    public let id: Int
    public let position: SCNVector3
    public let elevationDeg: Float
    public let azimuthDeg: Float
    public var isCaptured: Bool = false
    public var nodeSceneNode: SCNNode?
}

public class GeodesicDomePlanner {
    public private(set) var targetNodes: [TargetNode] = []
    public let center: SCNVector3
    public let radius: Float
    private let satisfactionAngleDeg: Float
    
    public init(center: SCNVector3, radius: Float = 1.5, numElevationRings: Int = 3, samplesPerRing: Int = 12, satisfactionAngleDeg: Float = 20.0) {
        self.center = center
        self.radius = radius
        self.satisfactionAngleDeg = satisfactionAngleDeg
        generateTargetNodes(numRings: numElevationRings, samplesPerRing: samplesPerRing)
    }
    
    private func generateTargetNodes(numRings: Int, samplesPerRing: Int) {
        targetNodes.removeAll()
        var nodeId = 0
        let elevations: [Float] = [15.0, 45.0, 75.0]
        
        for elev in elevations {
            let elevRad = elev * .pi / 180.0
            let azimuthOffset: Float = (nodeId % 2 == 0) ? 0.0 : (180.0 / Float(samplesPerRing))
            
            for i in 0..<samplesPerRing {
                let azimuthDeg = (Float(i) * (360.0 / Float(samplesPerRing)) + azimuthOffset).truncatingRemainder(dividingBy: 360.0)
                let azimuthRad = azimuthDeg * .pi / 180.0
                
                let x = center.x + radius * cos(elevRad) * cos(azimuthRad)
                let y = center.y + radius * sin(elevRad)
                let z = center.z + radius * cos(elevRad) * sin(azimuthRad)
                
                let nodePos = SCNVector3(x, y, z)
                let node = TargetNode(id: nodeId, position: nodePos, elevationDeg: elev, azimuthDeg: azimuthDeg)
                targetNodes.append(node)
                nodeId += 1
            }
        }
    }
    
    /// Evaluates current camera position relative to target nodes.
    /// Returns the closest node ID, its angular distance in degrees, and whether it was satisfied.
    public func evaluateCameraPose(cameraTransform: simd_float4x4) -> (closestNodeIndex: Int?, angularDistanceDeg: Float, isNewlyCaptured: Bool) {
        let camPos = SCNVector3(cameraTransform.columns.3.x, cameraTransform.columns.3.y, cameraTransform.columns.3.z)
        
        // Ray from object center to camera
        let camDir = simd_normalize(simd_float3(camPos.x - center.x, camPos.y - center.y, camPos.z - center.z))
        
        var minAngDeg: Float = 180.0
        var closestIdx: Int? = nil
        var newlyCaptured = false
        
        for i in 0..<targetNodes.count {
            let targetPos = targetNodes[i].position
            let targetDir = simd_normalize(simd_float3(targetPos.x - center.x, targetPos.y - center.y, targetPos.z - center.z))
            
            let dotProduct = simd_clamp(simd_dot(camDir, targetDir), -1.0, 1.0)
            let angRad = acos(dotProduct)
            let angDeg = angRad * 180.0 / .pi
            
            if angDeg < minAngDeg {
                minAngDeg = angDeg
                closestIdx = i
            }
            
            if angDeg <= satisfactionAngleDeg && !targetNodes[i].isCaptured {
                targetNodes[i].isCaptured = true
                newlyCaptured = true
            }
        }
        
        return (closestIdx, minAngDeg, newlyCaptured)
    }
    
    public var overallCoverageRatio: Float {
        guard !targetNodes.isEmpty else { return 0.0 }
        let captured = targetNodes.filter { $0.isCaptured }.count
        return Float(captured) / Float(targetNodes.count)
    }
}
