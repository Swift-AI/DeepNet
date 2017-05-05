//
//  Tensor.swift
//  DeepNet
//
//  Created by Collin Hundley on 5/4/17.
//
//

import Foundation
import Metal


@available(OSX 10.11, *)
public class Tensor {
    
    public var buffer: MTLBuffer?
    public var device: MTLDevice?
    var dimensions: [Int] = []
    
    public init(device: MTLDevice, dimensions: [Int], data: [Float]? = nil) {
        self.device = device
        self.dimensions = dimensions
        if let data = data {
            let length = MemoryLayout<Float>.size * data.count
            self.buffer = device.makeBuffer(bytes: data, length: length, options: .storageModeManaged)
        } else {
            self.buffer = device.makeBuffer(length: 0, options: .storageModeManaged)
        }
    }
    
    func write(_ data: UnsafeMutablePointer<Float>, count: Int) {
        // TODO: is it better to reset buffer or use buffer.contents().copyBytes?
        // We would also need to notify of new size, if necessary
        
        let length = MemoryLayout<Float>.size * count
        buffer = device?.makeBuffer(bytes: data, length: length, options: .storageModeManaged)
    }
    
    func write(_ array: [Float]) {
        // TODO: is it better to reset buffer or use buffer.contents().copyBytes?
        // We would also need to notify of new size, if necessary
        
        let length = MemoryLayout<Float>.size * array.count
        buffer = device?.makeBuffer(bytes: array, length: length, options: .storageModeManaged)
    }
    
    func readArray() -> [Float] {
        guard let buf = buffer else { return [] }
        let count = buf.length / MemoryLayout<Float>.size
        return Array(UnsafeBufferPointer(start: buf.contents()
            .assumingMemoryBound(to: Float.self), count: count))
    }
    
}


