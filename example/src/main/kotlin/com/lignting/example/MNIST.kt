package com.lignting.example

import java.io.File
import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import com.lignting.neural.Adam
import com.lignting.neural.CrossEntropy
import com.lignting.neural.Dense
import com.lignting.neural.ExponentialScheduler
import com.lignting.neural.LeakyRelu
import com.lignting.neural.Model
import com.lignting.neural.Relu
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.toList

fun main() {
    println("MNIST DataFrame Example")
    val trainDataPairs = loadMnistAsPairs("./data/mnist/mnist_train.csv")
    val testDataPairs = loadMnistAsPairs("./data/mnist/mnist_test.csv")
    println("Loaded ${trainDataPairs.size} training samples.")
    println("Loaded ${testDataPairs.size} testing samples.")
    
    val trainData = pairsToMatrices(trainDataPairs)
    val testData = pairsToMatrices(testDataPairs)
    println("Training data shapes: labels ${trainData.first.shape.contentToString()}, features ${trainData.second.shape.contentToString()}")
    println("Testing data shapes: labels ${testData.first.shape.contentToString()}, features ${testData.second.shape.contentToString()}")
    
    val model = Model(
        Dense(784, 128),
        LeakyRelu(),
        Dense(128, 64),
        LeakyRelu(),
        Dense(64, 10),
        loss = CrossEntropy(),
        optimizer = Adam(),
        scheduler = ExponentialScheduler(4e-2, dropRate = 1 - 1e-3)
    )
    val (trainX, trainY) = trainData
    val (testX, testY) = testData
    
    val epochs = 20
    for (epoch in 1..epochs) {
        val loss = model.fit(trainX, trainY, 1)
        val evaluate = model.evaluate(testX, testY)
        println("Epoch $epoch: Train Loss = $loss, Test Evaluate = $evaluate")
    }
    println("Training complete.")
    
    val predict = model.predict(testX)
    val accuracy = accuracy(predict, testY)
    println("Accuracy: $accuracy")
}

fun loadMnistAsPairs(path: String, hasHeader: Boolean = true): List<Pair<D1Array<Int>, D1Array<Int>>> {
    val rows: List<List<String>> = csvReader { delimiter = ',' }.readAll(File(path))
    val dataRows = if (hasHeader) rows.drop(1) else rows
    val result = ArrayList<Pair<D1Array<Int>, D1Array<Int>>>(dataRows.size)
    
    for (row in dataRows) {
        if (row.isEmpty()) continue
        val labelInt = row.getOrNull(0)?.toIntOrNull() ?: 0
        // one-hot 长度 10
        val labelArr = IntArray(10) { idx -> if (idx == labelInt) 1 else 0 }
        
        // 像素部分确保长度为 784（不足补 0，多余截断）
        val pixels = IntArray(784)
        for (i in 0 until 784) {
            val v = row.getOrNull(i + 1)
            pixels[i] = v?.toIntOrNull() ?: 0
        }
        
        val labelD1: D1Array<Int> = mk.ndarray(labelArr)
        val pixelsD1: D1Array<Int> = mk.ndarray(pixels)
        result.add(pixelsD1 to labelD1)
    }
    
    return result
}

fun pairsToMatrices(dataset: List<Pair<D1Array<Int>, D1Array<Int>>>): Pair<D2Array<Double>, D2Array<Double>> {
    val n = dataset.size
    if (n == 0) {
        // 返回空矩阵（形状 0 x 0）
        val emptyLabels = mk.ndarray(arrayOf(DoubleArray(0)))
        val emptyFeatures = mk.ndarray(arrayOf(DoubleArray(0)))
        return emptyLabels to emptyFeatures
    }
    
    val labelDim = dataset[0].first.size
    val featureDim = dataset[0].second.size
    
    // 校验所有样本的维度一致
    for ((lbl, feat) in dataset) {
        if (lbl.size != labelDim) throw IllegalArgumentException("Inconsistent label dimension: expected $labelDim, got ${lbl.size}")
        if (feat.size != featureDim) throw IllegalArgumentException("Inconsistent feature dimension: expected $featureDim, got ${feat.size}")
    }
    
    val labelsMat = dataset.map { it.first.toList().map { it.toDouble() } }.let { mk.ndarray(it) }
    val featuresMat = dataset.map { it.second.toList().map { it.toDouble() } }.let { mk.ndarray(it) }
    return labelsMat to featuresMat
}

fun accuracy(predictions: D2Array<Double>, labels: D2Array<Double>): Double {
    val pShape = predictions.shape
    val lShape = labels.shape
    if (pShape.size != 2 || lShape.size != 2) throw IllegalArgumentException("Only 2D arrays are supported")
    
    val transposedLabels = when {
        pShape[0] == lShape[0] && pShape[1] == lShape[1] -> false
        pShape[0] == lShape[1] && pShape[1] == lShape[0] -> true
        else -> throw IllegalArgumentException("Shape mismatch: predictions ${pShape.contentToString()} vs labels ${lShape.contentToString()}")
    }
    
    val rows = pShape[0]
    val cols = pShape[1]
    var correct = 0
    
    for (i in 0 until rows) {
        // argmax for prediction row i
        var pMaxIdx = 0
        var pMaxVal = predictions[i, 0]
        for (j in 1 until cols) {
            val v = predictions[i, j]
            if (v > pMaxVal) {
                pMaxVal = v
                pMaxIdx = j
            }
        }
        
        // argmax for label row i (consider possible transpose)
        var lMaxIdx = 0
        var lMaxVal = if (!transposedLabels) labels[i, 0] else labels[0, i]
        for (j in 1 until cols) {
            val v = if (!transposedLabels) labels[i, j] else labels[j, i]
            if (v > lMaxVal) {
                lMaxVal = v
                lMaxIdx = j
            }
        }
        
        if (pMaxIdx == lMaxIdx) correct++
    }
    
    return if (rows == 0) 0.0 else correct.toDouble() / rows.toDouble()
}