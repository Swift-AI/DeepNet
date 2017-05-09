//
//  DeepNet+Tensor.swift
//  DeepNet
//
//  Created by Collin Hundley on 5/7/17.
//
//

import Foundation
import Metal


public extension DeepNet {
    
    
    /// Generates a `Tensor` for use with `DeepNet`.
    ///
    /// - Parameters:
    ///   - dimensions: The dimensions of the underlying data.
    ///   - data: An array `[Float]` to copy into the new tensor's buffer.
    /// - Returns: A newly-allocated `Tensor`, populated with the data provided.
    public func makeTensor(dimensions: [Int], data: [Float]? = nil) throws -> Tensor {
        if let data = data {
            return try Tensor(device: device, dimensions: dimensions, data: data)
        } else {
            return Tensor(device: device, dimensions: dimensions)
        }
    }
    
}
