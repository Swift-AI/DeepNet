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
    
    /// The useful dimensions of the underlying data.
    /// This does **not** include any padding that has been applied.
    /// This may be of arbitrary rank.
    /// Examples:
    /// - `[1, 5]` is considered a row vector of length 5 (1 row, 5 colums).
    /// - `[10, 15]` is considered a 2D array with 10 rows and 15 columns.
    public var dimensions: [Int]
    
    /// The true dimensions of the underlying data.
    /// This **does** include padding, when applicable.
    public var paddedDimensions: [Int]
    
    /// The total number of useful data elements.
    /// This is the product of all entries in `dimensions`.
    /// This does **not** include any padding that has been applied.
    public var count: Int {
        return dimensions.reduce(1, *)
    }
    
    /// The total number of data elements, **including** padding.
    public var paddedCount: Int {
        return paddedDimensions.reduce(1, *)
    }
    
    /// The length of the underlying data, in bytes.
    /// This **does** include all padding, when applicable.
    public var length: Int {
        return MemoryLayout<Float>.stride * count
    }
    
    /// The Metal buffer containing all data.
    var buffer: MTLBuffer
    
    
    /// Do not initialize a `Tensor` manually. Instead, use `DeepNet.makeTensor()`.
    init(device: MTLDevice, dimensions: [Int], data: [Float], paddedTo padding: [Int] = [8, 8]) throws {
        self.dimensions = dimensions
        
        // Make sure data size matches dimensions
        let count = dimensions.reduce(1, *)
        guard data.count == count else {
            throw Tensor.Error.data("The provided data size does not match the specified dimensions. Data: \(data.count) bytes. Expected: \(count) bytes.")
        }
        
        // Pad data if needed
        let (paddedDimensions, paddedData) = Tensor.pad(data, dimensions: dimensions, to: padding)
        self.paddedDimensions = paddedDimensions
        let paddedLength = paddedDimensions.reduce(1, *) * MemoryLayout<Float>.stride
        self.buffer = device.makeBuffer(bytes: paddedData, length: paddedLength, options: .storageModeManaged)
    }
    
    
    /// Do not initialize a `Tensor` manually. Instead, use `DeepNet.makeTensor()`.
    init(device: MTLDevice, dimensions: [Int], paddedTo padding: [Int] = [8, 8]) {
        self.dimensions = dimensions
        let paddedDimensions = Tensor.paddedDimensions(for: dimensions, padding: padding)
        self.paddedDimensions = paddedDimensions
        let paddedCount = paddedDimensions.reduce(1, *)
        let length = paddedCount * MemoryLayout<Float>.stride
        // Note: We must fill with zeros so that padding doesn't produce NaN
        let zeros = [Float](repeatElement(0, count: paddedCount))
        self.buffer = device.makeBuffer(bytes: zeros, length: length, options: .storageModeManaged)
    }

}


// MARK: Reading and writing

public extension Tensor {
    
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
    
    
    /// Reads the useful data from the underlying storage buffer into an array, **not** including padding.
    ///
    /// - Returns: An array `[Float]` of the data contained by the receiver (without padding).
    public func read() -> [Float] {
        // Return full array if there is no padding
        if dimensions == paddedDimensions {
            let count = buffer.length / MemoryLayout<Float>.size
            // Read full array
            return Array(UnsafeBufferPointer(start: buffer.contents()
                .assumingMemoryBound(to: Float.self), count: count))
        }
        // Allocate output array
        var output = [Float](repeatElement(0, count: count))
        let ptr = buffer.contents().assumingMemoryBound(to: Float.self)
        // Populate output with all non-padding data
        for row in 0..<dimensions[0] {
            for col in 0..<paddedDimensions[1] {
                if col >= dimensions[1] { continue } // Skip padding columns
                let ptrIdx = row * paddedDimensions[1] + col
                let outputIdx = row * dimensions[1] + col
                output[outputIdx] = ptr[ptrIdx]
            }
        }
        return output
    }
    
    
    /// Reads all data from the underlying storage buffer - **including padding** - into an array.
    ///
    /// - Returns: An array `[Float]` of the data contained by the receiver.
    public func readPadded() -> [Float] {
        let count = buffer.length / MemoryLayout<Float>.size
        return Array(UnsafeBufferPointer(start: buffer.contents()
            .assumingMemoryBound(to: Float.self), count: count))
    }
    
}


// MARK: Padding

fileprivate extension Tensor {
    
    /// Calculates the dimensions of a matrix with the given input size, padded to the given dimensions.
    /// For example, an input size of `[6, 6]` and padding `[4, 4]` will produce a final size of `[8, 8]`.
    ///
    /// - Parameters:
    ///   - size: The original, non-padded matrix size.
    ///   - padding: The padding.
    /// - Returns: The dimensions of a new matrix with the padding applied.
    static func paddedDimensions(for size: [Int], padding: [Int]) -> [Int] {
        return [
            size[0] + (padding[0] - size[0] % padding[0]) % padding[0],
            size[1] + (padding[1] - size[1] % padding[1]) % padding[1]
        ]
    }
    
    
    /// Pads a matrix with zeros to achieve dimensions that are even multiples of the given size.
    /// For example, a matrix with dimensions `[6, 6]` padded to `[4, 4]` will receive 2 zero rows
    /// and to zero columns (on the bottom and right) to become size `[8, 8]`.
    ///
    /// - Parameters:
    ///   - matrix: The original matrix to pad.
    ///   - dimensions: The **current** dimensions of the matrix.
    ///   - to: The padding.
    /// - Returns: The new, padded matrix.
    static func pad(_ matrix: [Float], dimensions: [Int], to: [Int]) -> (dimensions: [Int], data: [Float]) {
        // Determine needed padding in each dimension
        let rows = dimensions[0]
        let cols = dimensions[1]
        let padRows = (to[0] - rows % to[0]) % to[0]
        let padCols = (to[1] - cols % to[1]) % to[1]
        let totalRows = rows + padRows
        let totalCols = cols + padCols
        
        // Return original matrix if the dimensions are already padded exactly
        if padRows == 0 && padCols == 0 {
            return (dimensions, matrix)
        }
        
        // Allocate space for full padded matrix
        var output = [Float](repeatElement(0, count: totalRows * totalCols))
        
        // Populate with original data
        for row in 0..<rows {
            for col in 0..<cols {
                let inIdx = row * cols + col
                let outIdx = row * totalCols + col
                output[outIdx] = matrix[inIdx]
            }
        }
        
        return ([totalRows, totalCols], output)
    }
    
}


// MARK: Debugging

public extension Tensor {
    
    public func print() {
        let matrix = read()
        for row in 0..<dimensions[0] {
            for col in 0..<dimensions[1] {
                let idx = row * dimensions[1] + col
                Swift.print(matrix[idx], separator: "", terminator: "\t")
            }
            Swift.print()
        }
    }
    
    
    public func printPadded() {
        let matrix = readPadded()
        for row in 0..<paddedDimensions[0] {
            for col in 0..<paddedDimensions[1] {
                let idx = row * paddedDimensions[1] + col
                Swift.print(matrix[idx], separator: "", terminator: "\t")
            }
            Swift.print()
        }
    }
    
}


