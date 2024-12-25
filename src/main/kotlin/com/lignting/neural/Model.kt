package com.lignting.neural

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import org.jetbrains.kotlinx.multik.ndarray.operations.toListD2

class Model(
    vararg layers: Layer,
    private val loss: Loss = Mse(),
    private val optimizer: Optimizer = GradientDescent()
) {
    private val layerList = layers.toList()
    fun fit(input: D2Array<Double>, output: D2Array<Double>): Double {
        val aList = mutableListOf(input)
        layerList.forEach { layer ->
            aList.add(layer.forward(aList.last()))
        }
        val lossResult = loss.loss(aList.last(), output)
        if (lossResult.isNaN()) {
            println("loss error")
            throw RuntimeException("loss error")
        }
        val dList = mutableListOf(loss.backward(aList.last(), output))
        layerList.zip(aList.dropLast(1)).asReversed().forEach { (layer, A) ->
            dList.add(
                layer.backward(
                    dList.last(),
                    A,
                    optimizer.copy()
                )
            )
        }
        return lossResult
    }

    fun fitWithBatchSize(input: D2Array<Double>, output: D2Array<Double>, batchSize: Int = 100): Double {
        val res = input.toListD2().zip(output.toListD2()).shuffled().chunked(batchSize).forEach {
            val input = mk.ndarray(it.map { it.first })
            val output = mk.ndarray(it.map { it.second })
            fit(input, output)
        }
        return evaluate(input, output)
    }

    fun predictOne(input: D1Array<Double>): D1Array<Double> {
        val aList = mutableListOf(input.reshape(1, input.shape[0]))
        layerList.forEach { layer ->
            aList.add(layer.forward(aList.last()))
        }
        return aList.last().let { it.reshape(it.shape[0] * it.shape[1]) }
    }

    fun predict(input: D2Array<Double>): D2Array<Double> {
        val aList = mutableListOf(input)
        layerList.forEach { layer ->
            aList.add(layer.forward(aList.last()))
        }
        return aList.last()
    }

    fun evaluate(input: D2Array<Double>, output: D2Array<Double>) = loss.loss(predict(input), output)

    fun copy() = Model(*layerList.map { it.copy() }.toTypedArray(), loss = loss, optimizer = optimizer.copy())
}