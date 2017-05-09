//
//  DeepNet+FullyConnected.swift
//  DeepNet
//
//  Created by Collin Hundley on 5/7/17.
//
//

import Foundation


// Convenience methods for creating and adding fully connected layers to DeepNet

public extension DeepNet {
    
    /// Creates and returns a new fully-connected layer.
    ///
    /// - Parameters:
    ///   - id: An ID to assign to the layer. If `nil`, a random ID will be generated.
    ///   - inputs: The number of inputs to the layer.
    ///   - outputs: The layer's number of outputs. Also the number of 'nodes' or 'neurons'.
    ///   - activation: An activation function to apply during forward propagation.
    /// - Returns: A new fully-connected layer.
    public func makeFullyConnectedLayer(id: String? = nil, inputs: Int, outputs: Int, activation: ActivationFunction,
                                        weights: [Float]? = nil, bias: [Float]? = nil) throws -> FullyConnectedLayer {
        return try FullyConnectedLayer(device: device, commandQueue: commandQueue, library: library,
                                       id: id, inputs: inputCount, batchSize: batchSize, outputs: outputs, activation: activation,
                                       weights: weights, bias: bias)
    }
    
    
    /// Adds a new fully-connected layer to the end of the graph.
    ///
    /// - Parameters:
    ///   - id: An ID to assign to the layer. If `nil`, a random ID will be generated.
    ///   - outputs: The layer's number of outputs. Also the number of 'nodes' or 'neurons'.
    ///   - activation: An activation function to apply during forward propagation.
    public func addFullyConnectedLayer(id: String? = nil, outputs: Int, activation: ActivationFunction,
                                       weights: [Float]? = nil, bias: [Float]? = nil) throws {
        let inputCount = layers.last?.outputSize ?? inputSize
        let layer = try FullyConnectedLayer(device: device, commandQueue: commandQueue, library: library,
                                            id: id, inputs: inputCount, batchSize: batchSize, outputs: outputs, activation: activation,
                                            weights: weights, bias: bias)
        addLayer(layer)
    }
    
    
    /// Inserts a new fully-connected layer into the graph,
    /// adjusting all parameters and caches automatically.
    ///
    /// - Parameters:
    ///   - id: An ID to assign to the layer. If `nil`, a random ID will be generated.
    ///   - outputs: The layer's number of outputs. Also the number of 'nodes' or 'neurons'.
    ///   - activation: An activation function to apply during forward propagation.
    ///   - index: The index at which to insert the layer in the graph.
    public func insertFullyConnectedLayer(id: String? = nil, outputs: Int, activation: ActivationFunction,
                                          weights: [Float]? = nil, bias: [Float]? = nil, atIndex index: Int) throws {
        guard index <= layers.count else {
            throw DeepNet.Error.layer("An invalid layer index has been provided for inserting.")
        }
        let inputCount = index == 0 ? inputSize : layers[index - 1].outputSize
        
        let layer = try FullyConnectedLayer(device: device, commandQueue: commandQueue, library: library,
                                            id: id, inputs: inputCount, batchSize: batchSize, outputs: outputs, activation: activation,
                                            weights: weights, bias: bias)
        try insertLayer(layer, at: index)
    }
    
}
