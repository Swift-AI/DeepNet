


// ------------ CODE FOR TESTING DeepNet DURING DEVELOPMENT ------------


import Foundation
import QuartzCore
import Metal


// Number of inputs to the neural network.
let inputCount = 65_536
let hiddenCount = 4096
let outputCount = 1000
/// Number of input sets per batch.
let batchSize = 8
/// Number of runs per test (to average execution times).
let runs = 5



// Prepare test data
print("Preparing data...")

// Allocate inputs
var inputs = [Float](repeatElement(0, count: inputCount * batchSize))
// Randomize values
for index in 0..<(inputCount * batchSize) {
    inputs[index] = Float(index)
//    inputs[index] = (Float(arc4random_uniform(2_000_000)) - 1_000_000) / 1_000_000 // (-1, 1)
}

let weights1 = [Float](repeatElement(1, count: inputCount * hiddenCount))
let bias1 = [Float](repeatElement(1, count: hiddenCount))
let weights2 = [Float](repeatElement(1, count: hiddenCount * outputCount))
let bias2 = [Float](repeatElement(1, count: outputCount))

//let weights3 = [Float](repeatElement(1, count: hiddenCount * hiddenCount))
//let bias3 = [Float](repeatElement(1, count: hiddenCount))

// Create neural net
print("Creating neural network...")

var net = DeepNet(inputs: inputCount)
net.batchSize = batchSize
try net.addFullyConnectedLayer(outputs: hiddenCount, activation: .tanh, weights: weights1, bias: bias1)
try net.addFullyConnectedLayer(outputs: outputCount, activation: .tanh, weights: weights2, bias: bias2)


// Run inference speed test
print("Running inference speed test...")

// Note: Here we use the 'out' variable to cause side effects in the loop, so the compiler doesn't optimize them away.
// We also log it at the end - the result is meaningless but ensures the loop was executed fully.

//func printOut(_ matrix: [Float], dimensions: [Int]) {
//    for row in 0..<dimensions[0] {
//        for col in 0..<dimensions[1] {
//            let idx = row * dimensions[1] + col
//            print(matrix[idx], separator: "", terminator: "\t")
//        }
//        print()
//    }
//}


var out: Float = 0
var avgTime: Double = 0
for _ in 0..<runs {
    let start = CACurrentMediaTime()
    let output = try! net.forward(inputs)
//    printOut(output, dimensions: [batchSize + 7, hiddenCount + 3])
    out += output[0] // Cause side effect
    let end = CACurrentMediaTime()
    avgTime += (end - start)
}
avgTime /= Double(runs)

print(out)
print("Done")


// Log time

print("\nAverage time per batch of \(batchSize): \(avgTime) seconds")



