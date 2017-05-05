//
//  Layer.swift
//  DeepNet
//
//  Created by Collin Hundley on 5/4/17.
//
//

import Foundation


@available(OSX 10.11, *)
public protocol Layer {
    
    var output: Tensor { get set }
    var weights: Tensor { get set }
    var bias: Tensor { get set }
    var activation: ActivationFunction { get set }
    
    /// Forward propagation.
    /// Accepts an input tensor and stores the result of the computation in `ourput`.
    func forward(input: Tensor)
    
}








