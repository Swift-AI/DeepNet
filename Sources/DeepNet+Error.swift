//
//  DeepNet+Error.swift
//  DeepNet
//
//  Created by Collin Hundley on 5/5/17.
//
//

import Foundation


public extension DeepNet {
    
    public enum Error: Swift.Error {
        /// Errors related to creating/inserting layers.
        case layer(String)
        /// Errors related to forward propagation.
        case forward(String)
        

        var localizedDescription: String {
            switch self {
            case .layer(let str):
                return str
            case .forward(let str):
                return str
            }
        }
    }
    
}

