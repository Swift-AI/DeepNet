//
//  Tensor.swift
//  DeepNet
//
//  Created by Collin Hundley on 5/4/17.
//
//

import Foundation
import Metal


// TODO: Provide subscript access to Tensor.
// Example: `tensor[3, 4]` would return the scalar at [3, 4] if `tensor` is of rank 2.
// `tensor[3]` would return the row vector at index 3 if `tensor` is of rank 2.
// Would it be possible to extract a column vector this way?
// How can we ensure type safetly while doing this? i.e. without just returning `Any`.

/// A multidimensional storage array backed by Metal.
public class Tensor {
    
    /// Errors.
    public enum Error: Swift.Error {
        case data(String)
        
        var localizedDescription: String {
            switch self {
            case .data(let str):
                return str
            }
        }
    }
    
    /// The dimensions of the underlying data.
    /// This may be of arbitrary rank.
    /// Examples:
    /// - `[5]` is considered a row vector of length 5 (1 row, 5 colums).
    /// - `[10, 15]` is considered a 2D array with 10 rows and 15 columns.
    public var dimensions: [Int]
    
    
    /// The total number of data elements.
    /// This is the product of all entries in `dimensions`.
    public var count: Int {
        return dimensions.reduce(1, *)
    }
    
    /// The length of the underlying data, in bytes.
    public var length: Int {
        return MemoryLayout<Float>.stride * count
    }
    
    /// The Metal buffer containing all data.
    var buffer: MTLBuffer
    
    
    /// Do not initialize a `Tensor` manually. Instead, use `DeepNet.makeTensor()`.
    init(device: MTLDevice, dimensions: [Int], data: [Float]) throws {
        self.dimensions = dimensions
        
        // Make sure data size matches dimensions
        let count = dimensions.reduce(1, *)
        let length = MemoryLayout<Float>.stride * count
        guard data.count == count else {
            throw Tensor.Error.data("The provided data size does not match the specified dimensions. Data: \(data.count) bytes. Expected: \(count) bytes.")
        }
        self.buffer = device.makeBuffer(bytes: data, length: length, options: .storageModeManaged)
    }
    
    
    /// Do not initialize a `Tensor` manually. Instead, use `DeepNet.makeTensor()`.
    init(device: MTLDevice, dimensions: [Int]) {
        self.dimensions = dimensions
        let length = MemoryLayout<Float>.stride * dimensions.reduce(1, *)
        self.buffer = device.makeBuffer(length: length, options: .storageModeManaged)
    }
    
    
    /// Copies the provided data into the receiver's underlying storage buffer.
    ///
    /// - Parameters:
    ///   - data: A pointer to data that shoudld be copied into the buffer.
    /// - Important: The size of `data` must equal the receiver's `length`.
    public func write(_ data: UnsafeMutablePointer<Float>) {
        buffer.contents().copyBytes(from: data, count: length)
        buffer.didModifyRange(NSMakeRange(0, length))
    }
    
    
    /// Copies the provided array into the receiver's underlying storage buffer.
    ///
    /// - Parameter array: An array `Float` to copy into the buffer.
    /// - Important: `array.count` must equal the receiver's `count`.
    /// - Throws: An error if a data inconsistency is encountered.
    public func write(_ array: [Float]) throws {
        guard array.count == count else {
            throw Tensor.Error.data("The size of the provided data does not match the receiver's size. Array count: \(array.count). Expected: \(count).")
        }
        buffer.contents().copyBytes(from: array, count: length)
        buffer.didModifyRange(NSMakeRange(0, length))
    }
    
    
    /// Reads the data from the underlying storage buffer into an array.
    ///
    /// - Returns: An array `[Float]` of the data contained by the receiver.
    public func read() -> [Float] {
        let count = buffer.length / MemoryLayout<Float>.size
        return Array(UnsafeBufferPointer(start: buffer.contents()
            .assumingMemoryBound(to: Float.self), count: count))
    }
    
}


