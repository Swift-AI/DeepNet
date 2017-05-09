//
//  FullyConnectedLayer.swift
//  DeepNet
//
//  Created by Collin Hundley on 5/5/17.
//
//

import Foundation
import Metal


struct MetalMatrixDimensions {
    let m: UInt32
    let k: UInt32
    let n: UInt32
    let pbytes: UInt32
    let qbytes: UInt32
}


public final class FullyConnectedLayer: Layer {
    
    // Properties
    
    /// A unique ID for this layer.
    public var id: String
    
    /// An activation function to apply to all elements in this layer.
    /// Changing the activation function is allowed, but will invalidate any training
    /// that has occurred.
    public var activation: ActivationFunction {
        didSet(old) {
            guard activation != old else { return }
            pipelineState = FullyConnectedLayer.pipelineStateForActivation(activation, device: device, library: library)
        }
    }
    
    /// A learning rate to apply during backpropagation.
    /// May be safely tuned at any time.
    public var learningRate: Float = 1.0
    
    /// A momentum factor to apply during backpropagation.
    /// May be safely tuned at any time.
    public var momentum: Float = 0
    
    
    // Caches
    
    /// The layer's cached output from the most recent forward pass.
    /// If a forward pass has not yet occurred, this Tensor will be zero-filled.
    public fileprivate(set) var output: Tensor
    
    /// The learned weights for this layer.
    fileprivate var weights: Tensor
    
    /// The learned biases for this layer.
    fileprivate var bias: Tensor
    
    
    // Metal
    
    /// The Metal device used for computations.
    public let device: MTLDevice
    
    /// The Metal library containing compiled GPU functions.
    public let library: MTLLibrary
    
    /// The Metal command queue for executing commands.
    public let commandQueue: MTLCommandQueue
    
    /// Reference to the Metal pipeline state containing the compiled functions needed for this layer.
    public fileprivate(set) var pipelineState: MTLComputePipelineState
    
    
    // Shape
    
    /// The number of expected inputs to the layer.
    public var inputSize: Int {
        didSet(old) {
            guard old != inputSize else { return }
            resetInputSize(inputSize)
        }
    }
    
    /// The number of sets per batch expected as inputs.
    public var batchSize: Int {
        didSet(old) {
            guard old != inputSize else { return }
            resetBatchSize(batchSize)
        }
    }
    
    /// The layer's number of outputs. Also the layer's number of 'nodes' or 'neurons'.
    public var outputSize: Int {
        didSet(old) {
            guard old != inputSize else { return }
            resetOutputSize(outputSize)
        }
    }
    
    
    // Initialization
    
    init(device: MTLDevice, commandQueue: MTLCommandQueue, library: MTLLibrary,
         id: String? = nil, inputs: Int, batchSize: Int, outputs: Int, activation: ActivationFunction,
         weights: [Float]? = nil, bias: [Float]? = nil) throws {
        self.id = id ?? UUID().uuidString // Generate random ID if none provided
        self.inputSize = inputs
        self.batchSize = batchSize
        self.outputSize = outputs
        self.activation = activation
        self.device = device
        self.library = library
        self.commandQueue = commandQueue
        self.pipelineState = FullyConnectedLayer.pipelineStateForActivation(activation, device: device, library: library)
        
        // Create empty output cache
        self.output = Tensor(device: device, dimensions: [batchSize, outputs])
        
        // Set weights
        if let weights = weights {
            self.weights = try Tensor(device: device, dimensions: [inputs, outputs], data: weights)
        } else {
            // Randomize initial weights
            let w = FullyConnectedLayer.generateRandomWeights(count: inputSize * outputSize,
                                                              inputs: inputSize,
                                                              outputs: outputSize,
                                                              activation: activation)
            self.weights = try Tensor(device: device, dimensions: [inputs, outputs], data: w)
        }
        
        // Set bias
        if let bias = bias {
            self.bias = try Tensor(device: device, dimensions: [1, outputs], data: bias) // Row vector
        } else {
            // Randomize initial biases
            let b = FullyConnectedLayer.generateRandomBiases(count: outputSize,
                                                             inputs: inputSize,
                                                             outputs: outputSize,
                                                             activation: activation)
            self.bias = try Tensor(device: device, dimensions: [1, outputs], data: b) // Row vector
        }
    }
    
    
    /// Compiles and returns a Metal pipeline state for the given activation function.
    private static func pipelineStateForActivation(_ activation: ActivationFunction,
                                                   device: MTLDevice, library: MTLLibrary) -> MTLComputePipelineState {
        switch activation {
        case .tanh:
            return try! device.makeComputePipelineState(function: library.makeFunction(name: "tanhForward")!)
        }
    }
    
}


// MARK: Forward propagation

public extension FullyConnectedLayer {
    
    /// Forward propagation.
    ///
    /// - Parameters:
    ///   - input: A `Tensor` containing data leading into the layer.
    public func forward(_ input: Tensor) throws {
        // Create command buffer + command encoder
        let commandBuffer = commandQueue.makeCommandBuffer()
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()
        commandEncoder.setComputePipelineState(pipelineState)
        
        // Set data buffers
        commandEncoder.setBuffer(input.buffer, offset: 0, at: 0)
        commandEncoder.setBuffer(weights.buffer, offset: 0, at: 1)
        commandEncoder.setBuffer(bias.buffer, offset: 0, at: 2)
        commandEncoder.setBuffer(output.buffer, offset: 0, at: 3)
        
        // Set dimensions buffer
        var dims = MetalMatrixDimensions(m: UInt32(input.dimensions[0]), // num batches
                                         k: UInt32(weights.dimensions[1]), // num outputs
                                         n: UInt32(input.dimensions[1]), // num inputs
                                         pbytes: UInt32(input.dimensions[1] * MemoryLayout<Float>.stride),
                                         qbytes: UInt32(weights.dimensions[1] * MemoryLayout<Float>.stride))
        let dimsBuffer = device.makeBuffer(bytes: &dims,
                                           length: MemoryLayout<MetalMatrixDimensions>.stride,
                                           options: .storageModeManaged)
        commandEncoder.setBuffer(dimsBuffer, offset: 0, at: 4)
        
        // Set threadgroup size
        let w = pipelineState.threadExecutionWidth
        // Note: The output matrix is more often wide (number of nodes) than tall (number of batches),
        // so we set width to max allowed and height to 1.
        let threadsPerThreadgroup = MTLSize(width: w, height: 1, depth: 1)
        // Note: Here we ensure that group height/width never equal zero (for output matrix with < 8 rows or columns)
        // We divide by 8 in both dimensions to account for the matrix multiplication kernel,
        // which operates on an 8x8 sector per thread.
        let minWidth = ceil(Double(weights.dimensions[1]) / 8)
        let groupWidth = (Int(minWidth) + w - 1) / w
        let groupHeight = Int(ceil(Double(input.dimensions[0]) / 8))
        let threadgroupsPerGrid = MTLSize(width: groupWidth,
                                          height: groupHeight,
                                          depth: 1)
        commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        commandEncoder.endEncoding()
        
        // Add blit command to synchronize output when finished
        let blitEncoder = commandBuffer.makeBlitCommandEncoder()
        blitEncoder.synchronize(resource: output.buffer)
        blitEncoder.endEncoding()
        
        // Execute
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
}


// MARK: Backpropagation

public extension FullyConnectedLayer {
    
    /// Performs backpropagation through the layer, updating all weights and biases as needed,
    /// based on the provided error gradient from the succeeding layer.
    /// This method is used for all hidden layers in the network.
    ///
    /// - Parameters:
    ///   - gradient: The error gradient of the succeeding layer, with respect to that layer's input.
    ///   - inputs: The inputs that were provided to this layer during the most recent forward pass.
    /// - Returns: This error gradient with respect to this layer's input.
    public func backpropagate(gradient: Tensor, inputs: Tensor) -> Tensor {
        
        
        return Tensor(device: device, dimensions: [])
    }
    
    
    /// Performs backpropagation through the layer, updating all weights and biases as needed,
    /// based on the provided target values (labels).
    /// This method is only used for the output layer in the network.
    ///
    /// - Parameters:
    ///   - target: The target output values (labels).
    ///   - inputs: The inputs that were provided to this layer during the most recent forward pass.
    /// - Returns: The error gradient with respect to this layer's input.
    public func backpropagate(target: Tensor, inputs: Tensor) -> Tensor {
        
        
        return Tensor(device: device, dimensions: [])
    }
    
}


// MARK: Reshaping + weights

public extension FullyConnectedLayer {

    fileprivate func resetInputSize(_ inputs: Int) {
        // Resize weights
        weights = Tensor(device: device, dimensions: [inputs, weights.dimensions[1]]) // Preserve output size
        // Re-randomize weights
        randomizeAllWeights()
    }
    
    
    fileprivate func resetBatchSize(_ size: Int) {
        // Resize output
        output = Tensor(device: device, dimensions: [size, output.dimensions[1]]) // Preserve output size
    }
    
    
    fileprivate func resetOutputSize(_ outputs: Int) {
        // Resize bias, weights and output
        bias = Tensor(device: device, dimensions: [outputs])
        weights = Tensor(device: device, dimensions: [weights.dimensions[0], outputs]) // Preserve input size
        output = Tensor(device: device, dimensions: [output.dimensions[0], outputs]) // Preserve batch size
        // Re-randomize all weights
        randomizeAllWeights()
        // Re-randomize all biases
        randomizeAllBiases()
    }
    
}


// MARK: Weights

public extension FullyConnectedLayer {
    
    /// Randomizes all weights for the layer, using an appropriate scaling factor for the current activation function.
    public func randomizeAllWeights() {
        let w = FullyConnectedLayer.generateRandomWeights(count: weights.buffer.length / MemoryLayout<Float>.stride,
                                                          inputs: inputSize, outputs: outputSize, activation: activation)
        weights.buffer.contents().copyBytes(from: w, count: weights.buffer.length)
        weights.buffer.didModifyRange(NSMakeRange(0, weights.buffer.length))
    }
    
    
    /// Randomizes all baises for the layer, using an appropriate scaling factor for the current activation function.
    public func randomizeAllBiases() {
        let b = FullyConnectedLayer.generateRandomBiases(count: bias.buffer.length / MemoryLayout<Float>.stride,
                                                         inputs: inputSize, outputs: outputSize, activation: activation)
        bias.buffer.contents().copyBytes(from: b, count: bias.buffer.length)
        bias.buffer.didModifyRange(NSMakeRange(0, bias.buffer.length))
    }
    
    
    /// Generates a set of random weights appropriate for the given layer size and activation function.
    fileprivate static func generateRandomWeights(count: Int, inputs: Int, outputs: Int, activation: ActivationFunction) -> [Float] {
        var w = [Float](repeatElement(0, count: count))
        for index in 0..<w.count {
            w[index] = randomWeight(fanIn: inputs, fanOut: outputs, activation: activation)
        }
        return w
    }
    
    
    //// Generates a set of random biases appropriate for the given layer size and activation function.
    fileprivate static func generateRandomBiases(count: Int, inputs: Int, outputs: Int, activation: ActivationFunction) -> [Float] {
        var b = [Float](repeatElement(0, count: count))
        for index in 0..<b.count {
            b[index] = randomWeight(fanIn: inputs, fanOut: outputs, activation: activation)
        }
        return b
    }
    
}


// MARK: Utilities

fileprivate extension FullyConnectedLayer {
    
    /// Generates a single random weight.
    ///
    /// - Parameter fanIn: The number of inputs to the node in which this weight will be used.
    /// - Parameter fanOut: The number of outputs from the node in which this weight will be used.
    /// - Parameter activation: The activation function used for this layer.
    /// - Returns: A randomly-generated weight optimized for this node and the network's hidden activation function.
    static func randomWeight(fanIn: Int, fanOut: Int, activation: ActivationFunction) -> Float {
        func scale(_ n: UInt32, min: Float, max: Float) -> Float {
            return (max - min) * Float(n) / Float(UInt32.max) + min
        }
        // sqrt(6 / (fanOut + fanIn))
        let range = sqrt(6 / Float(fanIn + fanOut)) / 2
        let rand = arc4random()
        
        switch activation {
//        case .sigmoid:
//            // 4 * sqrt(6 / (fanOut + fanIn))
//            return scale(rand, min: -range, max: range) * 4
        case .tanh:
            // sqrt(6 / (fanOut + fanIn))
            return scale(rand, min: -range, max: range)
        }
    }
    
}

