//
//  DeepNet.swift
//  DeepNet
//
//  Created by Collin Hundley on 5/4/17.
//
//

import Foundation
import Metal


public final class DeepNet {
    
    // Properties
    
    /// The number of inputs to the graph.
    public var inputSize: Int {
        didSet(old) {
            guard inputSize != old else { return }
            resetInputSize(inputSize)
        }
    }
    
    /// The expected number of items per batch for forward propagation.
    /// Default 1.
    public var batchSize: Int {
        didSet(old) {
            guard batchSize != old else { return }
            resetBatchSize(batchSize)
        }
    }
    
    /// All layers, in order of forward execution, currently contained in the graph.
    var layers: [Layer] = []
    
    
    // Metal 
    
    /// The Metal device for performing all GPU-bound computations.
    let device: MTLDevice
    
    /// The Metal command queue for executing commands.
    let commandQueue: MTLCommandQueue
    
    /// The Metal library containing all compiled GPU kernels.
    let library: MTLLibrary
    
    
    // Initialization
    
    public init(inputs: Int = 1) {
        // Prepare Metal
        (device, commandQueue, library) = DeepNet.prepareMetal()
        
        // Init properties
        self.inputSize = inputs
        self.batchSize = 1
    }
    
}


// MARK: Forward propagation

public extension DeepNet {
    
    @discardableResult
    public func forward(_ input: Tensor) throws -> [Float] {
        // Make sure the network is prepared for inference
        guard isReadyToForward() else {
            throw DeepNet.Error.forward("The graph is not prepared for forward propagation. Make sure you've added at least one layer.")
        }
        
        // Make sure the input size matches our input/batch size
        guard input.count == inputSize * batchSize else {
            throw DeepNet.Error.forward("An incorrect number of inputs was provided: \(input.count). Expected: \(inputSize * batchSize).")
        }
        
        // Forward results through each layer
        try layers[0].forward(input)
        for i in 1..<layers.count {
            try layers[i].forward(layers[i - 1].output)
        }
        
        // Return last output
        return layers.last!.output.read()
    }
    
    @discardableResult
    public func forward(_ input: [Float]) throws -> [Float] {
        // Create a Tensor from the input data
        let inTensor = try makeTensor(dimensions: [batchSize, inputSize], data: input)
        return try forward(inTensor)
    }
    
    @discardableResult
    public func forward(_ input: UnsafeBufferPointer<Float>) throws -> [Float] {
        
        // TODO
        
        return []
    }
    
    public func forward(_ input: UnsafeBufferPointer<Float>, output: UnsafePointer<Float>) throws {
        
        // TODO
        
    }
    
    /// Returns `true` if the graph is prepared for forward propagation.
    private func isReadyToForward() -> Bool {
        return !layers.isEmpty
    }
    
}


// MARK: Adding/inserting layers

public extension DeepNet {
    

    /// Adds a new layer to the end of the graph, adjusting all parameters and caches automatically.
    ///
    /// - Parameter layer: The layer to add. This becomes the graph's output.
    public func addLayer(_ layer: Layer) {
        if layers.isEmpty {
            layer.inputSize = inputSize
        } else {
            let previousLayerSize = layers[layers.count - 1].outputSize
            layer.inputSize = previousLayerSize
        }
        layers.append(layer)
    }
    
    
    /// Inserts a new layer into the graph, adjusting all parameters and caches automatically.
    ///
    /// - Parameters:
    ///   - layer: The layer to insert. If `index == layers.count`, this layer becomes the neural network's output.
    ///   - index: The index to insert the graph in the network.
    ///
    /// - Note: If sizing conflicts occur, this may invalidate any training that has occurred
    ///         for this layer and the layer immediately following it.
    public func insertLayer(_ layer: Layer, at index: Int) throws {
        guard index <= layers.count else {
            throw DeepNet.Error.layer("An invalid layer index has been provided for inserting.")
        }
        if index == layers.count {
            return addLayer(layer)
        }
        let inputCount = index == 0 ? inputSize : layers[index - 1].outputSize
        layers.insert(layer, at: index)
        layer.inputSize = inputCount
        layers[index + 1].inputSize = layer.outputSize
    }
    
    
    /// Resets the entire graph with the given layers.
    /// The user is responsible for ensuring consistency in sizing between layers.
    ///
    ///
    /// - Parameter layers: The layers to build the graph.
    public func setLayers(_ layers: [Layer]) {
        self.layers = layers
    }

}


// MARK: Resizing

fileprivate extension DeepNet {
    
    /// Resets the expected input size for the graph.
    ///
    /// - Parameter size: The number of elements per input set.
    func resetInputSize(_ size: Int) {
        guard let layer = layers.first else { return }
        layer.inputSize = size
    }
    
    
    /// Resets the input batch size for the graph.
    ///
    /// - Parameter size: The number of input sets per batch.
    func resetBatchSize(_ size: Int) {
        for layer in layers {
            layer.batchSize = size
        }
    }
    
}


// MARK: Utilities

extension DeepNet {
    
    fileprivate static func prepareMetal() -> (device: MTLDevice, commandQueue: MTLCommandQueue, library: MTLLibrary) {
        let device = MTLCreateSystemDefaultDevice()!
        let commandQueue = device.makeCommandQueue()
        let library = device.newDefaultLibrary()!
        return (device, commandQueue, library)
    }
    
}



