//
//  ARCameraGuideViewController.swift
//  3DGS Camera Guide (iOS ARKit)
//
//  Main iOS View Controller integrating ARKit 6-DoF SLAM tracking,
//  SceneKit 3D geodesic dome target nodes, haptic feedback, and dataset export.
//

import UIKit
import ARKit
import SceneKit

public class ARCameraGuideViewController: UIViewController, ARSCNViewDelegate, ARSessionDelegate {
    
    // UI Elements
    private let sceneView = ARSCNView()
    private let statusLabel = UILabel()
    private let progressBar = UIProgressView(progressViewStyle: .default)
    private let captureButton = UIButton(type: .system)
    private let exportButton = UIButton(type: .system)
    
    // Core Logic Managers
    private var domePlanner: GeodesicDomePlanner?
    private let datasetExporter = DatasetExporter()
    private let feedbackGenerator = UIImpactFeedbackGenerator(style: .medium)
    
    private var objectCenterWorldPos: SCNVector3?
    private var isSessionInitialized = false
    private var nodeSceneNodes: [SCNNode] = []
    
    override public func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        setupARScene()
        feedbackGenerator.prepare()
    }
    
    override public func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = [.horizontal, .vertical]
        configuration.isAutoFocusEnabled = true
        sceneView.session.run(configuration)
    }
    
    override public func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        sceneView.session.pause()
    }
    
    // MARK: - UI Setup
    private func setupUI() {
        view.addSubview(sceneView)
        sceneView.frame = view.bounds
        sceneView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        sceneView.delegate = self
        sceneView.session.delegate = self
        
        // Tap Gesture to set Object Center
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(handleTapToPlaceCenter(_:)))
        sceneView.addGestureRecognizer(tapGesture)
        
        // Status Bar Label
        statusLabel.text = "Tap on object to place 3D Target Orbit"
        statusLabel.textColor = .white
        statusLabel.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        statusLabel.textAlignment = .center
        statusLabel.font = UIFont.systemFont(ofSize: 16, weight: .bold)
        statusLabel.layer.cornerRadius = 8
        statusLabel.clipsToBounds = true
        statusLabel.frame = CGRect(x: 20, y: 60, width: view.bounds.width - 40, height: 44)
        view.addSubview(statusLabel)
        
        // Progress Bar
        progressBar.frame = CGRect(x: 20, y: 110, width: view.bounds.width - 40, height: 10)
        progressBar.progress = 0.0
        progressBar.tintColor = .systemGreen
        view.addSubview(progressBar)
        
        // Manual Capture Button
        captureButton.setTitle("📸 Auto-Capture Target", for: .normal)
        captureButton.setTitleColor(.white, for: .normal)
        captureButton.backgroundColor = .systemBlue
        captureButton.layer.cornerRadius = 25
        captureButton.frame = CGRect(x: (view.bounds.width - 220) / 2, y: view.bounds.height - 120, width: 220, height: 50)
        captureButton.addTarget(self, action: #selector(captureCurrentFrame), for: .touchUpInside)
        view.addSubview(captureButton)
    }
    
    private func setupARScene() {
        sceneView.autoenablesDefaultLighting = true
        sceneView.showsStatistics = false
    }
    
    // MARK: - Tap to Set Object Bounding Center
    @objc private func handleTapToPlaceCenter(_ gesture: UITapGestureRecognizer) {
        let touchLocation = gesture.location(in: sceneView)
        guard let raycastQuery = sceneView.raycastQuery(from: touchLocation, allowing: .estimatedPlane, alignment: .any) else { return }
        
        let results = sceneView.session.raycast(raycastQuery)
        guard let firstResult = results.first else { return }
        
        let worldTransform = firstResult.worldTransform
        let centerPos = SCNVector3(worldTransform.columns.3.x, worldTransform.columns.3.y, worldTransform.columns.3.z)
        
        self.objectCenterWorldPos = centerPos
        self.domePlanner = GeodesicDomePlanner(center: centerPos, radius: 1.2, numElevationRings: 3, samplesPerRing: 12)
        
        spawn3DTargetDomeNodes()
        isSessionInitialized = true
        
        statusLabel.text = "Target Dome Set! Orbit object to capture."
        feedbackGenerator.impactOccurred()
    }
    
    // MARK: - Render 3D Target Nodes in AR
    private func spawn3DTargetDomeNodes() {
        nodeSceneNodes.forEach { $0.removeFromParentNode() }
        nodeSceneNodes.removeAll()
        
        guard let planner = domePlanner else { return }
        
        for node in planner.targetNodes {
            let sphereGeometry = SCNSphere(radius: 0.03)
            let material = SCNMaterial()
            material.diffuse.contents = UIColor.systemRed.withAlphaComponent(0.8)
            sphereGeometry.materials = [material]
            
            let scnNode = SCNNode(geometry: sphereGeometry)
            scnNode.position = node.position
            sceneView.scene.rootNode.addChildNode(scnNode)
            nodeSceneNodes.append(scnNode)
        }
    }
    
    // MARK: - Real-Time 60 FPS Session Delegate
    public func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard isSessionInitialized, let planner = domePlanner else { return }
        
        let cameraTransform = frame.camera.transform
        let evalResult = planner.evaluateCameraPose(cameraTransform: cameraTransform)
        
        // Update Progress UI
        let ratio = planner.overallCoverageRatio
        DispatchQueue.main.async {
            self.progressBar.progress = ratio
            if let idx = evalResult.closestNodeIndex {
                let dist = evalResult.angularDistanceDeg
                self.statusLabel.text = String(format: "Coverage: %.0f%% | Target Gap: %.1f°", ratio * 100, dist)
            }
        }
        
        // If camera reached an unsatisfied target node (< 20° gap) -> Trigger Auto-Capture!
        if evalResult.isNewlyCaptured, let idx = evalResult.closestNodeIndex {
            DispatchQueue.main.async {
                self.updateNodeColorToGreen(index: idx)
                self.captureCurrentFrame()
            }
        }
    }
    
    private func updateNodeColorToGreen(index: Int) {
        guard index < nodeSceneNodes.count else { return }
        let node = nodeSceneNodes[index]
        node.geometry?.firstMaterial?.diffuse.contents = UIColor.systemGreen
        feedbackGenerator.impactOccurred()
    }
    
    // MARK: - Capture Frame Quality Check & Export
    @objc private func captureCurrentFrame() {
        guard let currentFrame = sceneView.session.currentFrame else { return }
        let pixelBuffer = currentFrame.capturedImage
        
        // Check Blur Variance
        let sharpnessScore = BlurDetector.computeLaplacianVariance(pixelBuffer: pixelBuffer)
        if sharpnessScore < 80.0 {
            statusLabel.text = "⚠️ Motion Blur Detected! Hold phone steady."
            return
        }
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }
        let uiImage = UIImage(cgImage: cgImage)
        
        datasetExporter.saveCapturedFrame(image: uiImage, camera: currentFrame.camera, sharpnessScore: sharpnessScore)
        feedbackGenerator.impactOccurred()
    }
}
