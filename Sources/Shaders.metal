//
//  Shaders.metal
//  DeepNet
//
//  Created by Collin Hundley on 5/5/17.
//
//

#include <metal_stdlib>
using namespace metal;


// MARK: Matrix Multiplication

/// A set of dimensions required for performing matrix multiplication.
///
/// - m: The number of rows in matrices A and C.
///      This should be the number of rows that you actually wish to compute (not including padding).
/// - k: The number of columns in matrices B and C.
///      This should be the number of columns that you actually wish to compute (not including padding).
/// - n: The number of columns in matrix A; number of rows in matrix B.
///      This should be the number of columns that you actually wish to compute (not including padding).
/// - pbytes: The stride in bytes from one row to another in matrix A (including padding).
///           Must be a multiple of 32. A should be padded to achieve this.
/// - qbytes: The stride in bytes from one row to another in matrix B (including padding).
typedef struct {
    uint m;
    uint k;
    uint n;
    uint pbytes;
    uint qbytes;
} MatrixDimensions;


/// Computes A * B and stores the result in C.
///
/// IMPORTANT: Each thread computes an 8x8 sector of the output C.
/// The rows and columns of each matrix must be padded to a multiple of 8.
/// So, threads should be only be dispatched for every 8th element of C in both dimensions.
///
/// Note: Here, `gidIn` must be the thread's absolute starting position in C.
/// i.e. the absolute matrix position in C of element (0, 0) for this thread.
/// It may be calculated like this: `uint2 gidIn = uint2(gid.x << 3, gid.y << 3);`
inline void matrixMultiply(const device float* A,
                           const device float* B,
                           device float* C,
                           constant MatrixDimensions& dims,
                           uint2 gidIn) {
    
//    uint n = dims.n;
    
//    uint pbytes = dims.pbytes;
//    uint qbytes = dims.qbytes;
    
    int pbytes = (int)dims.pbytes;
    int qbytes = (int)dims.qbytes;
    
    for (int row = 0; row < 8; row++) {
        int aStart = (gidIn.y + row) * pbytes / 4;
        for (int col = 0; col < 8; col++) {
            int cIdx = gidIn.y + (row * qbytes / 4) + gidIn.x + col;
            int bStart = gidIn.x + col;
            
            float sum = 0.0f;
            for (int i = 0; i < (pbytes / 4); i++) {
                float prod = A[aStart + i] * B[bStart + (i * qbytes / 4)];
                sum += prod;
            }
            
            C[cIdx] = sum;
            
        }
    }
    
//    const device float4* a = (const device float4*)(A + gidIn.x);
//    const device float4* b = (const device float4*)(B + gidIn.y);
//    
//    C = (device float*)((device char*)C + gidIn.x*qbytes);
//    
//    device float4* c = (device float4*)(C + gidIn.y);
//    
//    const device float4* Bend = (const device float4*)((const device char*)B + qbytes * n);
//    
//    float4 s0  = 0.0f, s1  = 0.0f, s2  = 0.0f, s3  = 0.0f;
//    float4 s4  = 0.0f, s5  = 0.0f, s6  = 0.0f, s7  = 0.0f;
//    float4 s8  = 0.0f, s9  = 0.0f, s10 = 0.0f, s11 = 0.0f;
//    float4 s12 = 0.0f, s13 = 0.0f, s14 = 0.0f, s15 = 0.0f;
//    
//    do {
//        float4 aCurr0 = a[0];
//        float4 aCurr1 = a[1];
//        float4 bCurr0 = b[0];
//        float4 bCurr1 = b[1];
//        
//        s0   += (aCurr0.x * bCurr0);
//        s2   += (aCurr0.y * bCurr0);
//        s4   += (aCurr0.z * bCurr0);
//        s6   += (aCurr0.w * bCurr0);
//        
//        s1   += (aCurr0.x * bCurr1);
//        s3   += (aCurr0.y * bCurr1);
//        s5   += (aCurr0.z * bCurr1);
//        s7   += (aCurr0.w * bCurr1);
//        
//        s8   += (aCurr1.x * bCurr0);
//        s10  += (aCurr1.y * bCurr0);
//        s12  += (aCurr1.z * bCurr0);
//        s14  += (aCurr1.w * bCurr0);
//        
//        s9   += (aCurr1.x * bCurr1);
//        s11  += (aCurr1.y * bCurr1);
//        s13  += (aCurr1.z * bCurr1);
//        s15  += (aCurr1.w * bCurr1);
//        
//        a = (device float4*)((device char*)a + pbytes);
//        b = (device float4*)((device char*)b + qbytes);
//        
//    } while(b < Bend);
//    
//    c[0] = s0;  c[1] = s1;  c = (device float4*)((device char*)c + qbytes);
//    c[0] = s2;  c[1] = s3;  c = (device float4*)((device char*)c + qbytes);
//    c[0] = s4;  c[1] = s5;  c = (device float4*)((device char*)c + qbytes);
//    c[0] = s6;  c[1] = s7;  c = (device float4*)((device char*)c + qbytes);
//    c[0] = s8;  c[1] = s9;  c = (device float4*)((device char*)c + qbytes);
//    c[0] = s10; c[1] = s11; c = (device float4*)((device char*)c + qbytes);
//    c[0] = s12; c[1] = s13; c = (device float4*)((device char*)c + qbytes);
//    c[0] = s14; c[1] = s15;
    
}


// MARK: Bias Addition

/// Adds a bias vector across each row in a matrix.
/// Executes on an 8x8 sector of the matrix.
inline void addBias(const device float* bi,
                    device float* C,
                    const int rows,
                    const int cols,
                    const int width,
                    uint2 gidIn) {

    int x = (int)gidIn.x;
    int y = (int)gidIn.y;
    
    // Add bias row vector to each row in C
    for (int row = 0; row < 8; row++) {
        if (y + row >= rows) break; // Don't add bias to padding rows
        for (int col = 0; col < 8; col++) {
            if (x + col >= cols) continue; // Don't add bias to padding columns
            int idx = (gidIn.y + row) * width + gidIn.x + col;
            C[idx] = C[idx] + bi[gidIn.x + col];
        }
    }
}


// MARK: Fully Connected Forward - Hyperbolic Tangent

/// Each thread of this kernel operates on an 8x8 sector of the
/// output buffer. Matrix buffers require padding to accomodate this.
///
/// Requirements:
///
/// The bytes-per-row of each matrix buffer must be padded to a multiple
/// of eight floats (32 bytes). Similarly, the row count of each matrix
/// must be padded to a multiple of eight.
kernel void tanhForward(const device float* input [[ buffer(0) ]],
                        const device float* weights [[ buffer(1) ]],
                        const device float* bi [[ buffer(2) ]], // bias
                        device float* output [[ buffer(3) ]],
                        constant MatrixDimensions& dims [[ buffer(4) ]],
                        uint2 gid [[ thread_position_in_grid ]]) {
    
    // Store sizes
    uint m = dims.m;
    uint k = dims.k;
    uint qbytes = dims.qbytes;
    
    // Determine and store absolute starting position
    // i.e. the absolute matrix position in C of element (0, 0) for this thread
    uint2 gidIn = uint2(gid.x << 3, gid.y << 3);
    
    // Return if out of bounds
    if (gidIn.x >= k || gidIn.y >= m) return;
    
    // Calculate full width of C - including any padding
    int width = qbytes >> 2;

    // Multiply input by weights
    matrixMultiply(input, weights, output, dims, gidIn);
    
    // Add bias vector across each row
    addBias(bi, output, m, k, width, gidIn);
    
    // Apply hyperbolic tangent activation to each element
    for (int row = 0; row < 8; row++) {
        if (gidIn.y + row >= m) continue; // Don't apply activation to padding rows
        for (int col = 0; col < 8; col++) {
            if (gidIn.x + col >= k) continue; // Don't apply activation to padding columns
            int idx = (gidIn.y + row) * width + gidIn.x + col;
            // Must clamp range to (-15, 15) to avoid tanh overflow while using fast math
            output[idx] = tanh(clamp(output[idx], -15.0f, 15.0f));
        }
    }
}







