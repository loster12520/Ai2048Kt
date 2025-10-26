package com.lignting.example

import java.io.File
import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import com.lignting.neural.*
import org.jetbrains.kotlinx.dataframe.api.dataFrameOf
import org.jetbrains.kotlinx.kandy.dsl.plot
import org.jetbrains.kotlinx.kandy.letsplot.export.save
import org.jetbrains.kotlinx.kandy.letsplot.layers.line
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.toArray
import org.jetbrains.kotlinx.multik.ndarray.operations.toList

fun main() {
    println("MNIST DataFrame Example")
    // 请从https://github.com/phoebetronic/mnist中获取这两个csv，并放置在./data/mnist/目录下
    val trainDataPairs = loadMnistAsPairs("./data/mnist/mnist_train.csv")
//        .let { it.filterIndexed { index, _ -> index < it.size * 0.1 } }
    val testDataPairs = loadMnistAsPairs("./data/mnist/mnist_test.csv")
    println("Loaded ${trainDataPairs.size} training samples.")
    println("Loaded ${testDataPairs.size} testing samples.")

    val trainData = pairsToMatrices(trainDataPairs)
    val testData = pairsToMatrices(testDataPairs)
    println("Training data shapes: features ${trainData.first.shape.contentToString()}, labels ${trainData.second.shape.contentToString()}")
    println("Testing data shapes: features ${testData.first.shape.contentToString()}, labels ${testData.second.shape.contentToString()}")

    val model = Model(
        Dense(784, 40, HeUniformInitialize()),
        LeakyRelu(),
        Dropout(0.2),
        Dense(40, 20, HeUniformInitialize()),
        LeakyRelu(),
        Dropout(0.2),
        Dense(20, 10, HeUniformInitialize()),
        Softmax(),
        loss = CrossEntropy(),
        optimizer = Adam(),
        scheduler = ExponentialScheduler(1e-4, dropRate = 1.0 - 1e-2)
    )
    val (trainX, trainY) = trainData
    val (testX, testY) = testData

    val epochs = 100
    val (lossList, evaluateList) = (0..epochs).map { epoch ->
        val loss = model.fitWithBatchSize(trainX, trainY, epoch, 100)
        val evaluate = model.evaluate(testX, testY)
        println("Epoch $epoch: Train Loss = $loss, Test Evaluate = $evaluate")
        loss to evaluate
    }.let {
        it.map { it.first } to it.map { it.second }
    }
    println("Training complete.")

    // 可视化结果
    val dataframe = dataFrameOf(
        "epochs" to List(lossList.size) { index -> index } + List(evaluateList.size) { index -> index },
        "loss" to lossList + evaluateList,
        "category" to lossList.map { "train" } + evaluateList.map { "test" }
    )
    val plot = dataframe.plot {
        line {
            x("epochs")
            y("loss")
            color("category")
        }
    }
    plot.save("mnist_loss_plot.png")

    val predict = model.predict(testX)
    println(
        "predict 3 samples: ${predict.toArray().slice(0 until 3).map { it.toList() }}, actual: ${
            testY.toArray().slice(0 until 3).map { it.toList() }
        }"
    )
    val accuracy = accuracy(predict, testY)
    println("Accuracy: $accuracy")
}

fun loadMnistAsPairs(path: String, hasHeader: Boolean = true): List<Pair<D1Array<Double>, D1Array<Double>>> {
    val rows: List<List<String>> = csvReader { delimiter = ',' }.readAll(File(path))
    val dataRows = if (hasHeader) rows.drop(1) else rows
    val result = ArrayList<Pair<D1Array<Double>, D1Array<Double>>>(dataRows.size)

    for (row in dataRows) {
        if (row.isEmpty()) continue
        val labelInt = row.getOrNull(0)?.toIntOrNull() ?: 0
        val labelArr = DoubleArray(10) { idx -> if (idx == labelInt) 1.0 else 0.0 }

        val pixels = DoubleArray(784) { i ->
            val v = row.getOrNull(i + 1)
            val raw = v?.toDoubleOrNull() ?: 0.0
            raw / 255.0
        }

        val labelD1: D1Array<Double> = mk.ndarray(labelArr)
        val pixelsD1: D1Array<Double> = mk.ndarray(pixels)
        result.add(pixelsD1 to labelD1)
    }

    return result
}

fun pairsToMatrices(dataset: List<Pair<D1Array<Double>, D1Array<Double>>>): Pair<D2Array<Double>, D2Array<Double>> {
    val n = dataset.size
    if (n == 0) {
        val empty = mk.ndarray(arrayOf(DoubleArray(0)))
        return empty to empty
    }

    val featureDim = dataset[0].first.size
    val labelDim = dataset[0].second.size

    for ((feat, lbl) in dataset) {
        if (feat.size != featureDim) throw IllegalArgumentException("Inconsistent feature dimension: expected $featureDim, got ${feat.size}")
        if (lbl.size != labelDim) throw IllegalArgumentException("Inconsistent label dimension: expected $labelDim, got ${lbl.size}")
    }

    val featuresMat = dataset.map { it.first.toList() }.let { mk.ndarray(it) }
    val labelsMat = dataset.map { it.second.toList() }.let { mk.ndarray(it) }
    return featuresMat to labelsMat
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