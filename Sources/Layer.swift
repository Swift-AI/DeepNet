//
//  Layer.swift
//  DeepNet
//
//  Created by Collin Hundley on 5/4/17.
//
//

import Foundation
import Metal


/// Protocol which all layers must adopt.
public protocol Layer: class {
    
    // Basic properties
    var id: String { get set }
    var activation: ActivationFunction { get set }
    
    // Shape
    var inputSize: Int { get set }
    var batchSize: Int { get set }
    var outputSize: Int { get set }
    
    // Training properties
    var learningRate: Float { get set }
    var momentum: Float { get set }
    
    // Cache
    var output: Tensor { get }
    
    // Metal
    var device: MTLDevice { get }
    var library: MTLLibrary { get }
    var commandQueue: MTLCommandQueue { get }
    var pipelineState: MTLComputePipelineState { get }
    
    // Forward propagation
    /// Forward propagation.
    /// Accepts an input tensor and stores the result of the computation in `output`.
    func forward(_ input: Tensor) throws
    
    // Backpropagation
    /// Performs backpropagation through the layer, updating all weights and biases as needed.
    ///
    /// - Parameters:
    ///   - gradient: The error gradient of the succeeding layer, with respect to that layer's input.
    ///   - inputs: The inputs that were provided to this layer during the most recent forward pass.
    /// - Returns: This error gradient with respect to this layer's input.
    func backpropagate(gradient: Tensor, inputs: Tensor) -> Tensor
    
    /// Performs backpropagation through the layer, updating all weights and biases as needed,
    /// based on the provided target values (labels).
    /// This method is only used for the output layer in the network.
    ///
    /// - Parameters:
    ///   - target: The target output values (labels).
    ///   - inputs: The inputs that were provided to this layer during the most recent forward pass.
    /// - Returns: The error gradient with respect to this layer's input.
    func backpropagate(target: Tensor, inputs: Tensor) -> Tensor
    
}
