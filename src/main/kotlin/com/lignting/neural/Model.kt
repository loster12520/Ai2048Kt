package com.lignting.neural

import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array

class Model(vararg layers: Layer, private val loss: Loss = Mse(), private val learningRate: Double = 0.01) {
    private val layerList = layers.toList()
    fun fit(input: D2Array<Double>, output: D2Array<Double>): Double {
        val AList = mutableListOf(input)
        layerList.forEach { layer ->
            AList.add(layer.forward(AList.last()))
        }
        val lossResult = loss.loss(AList.last(), output)
        if (lossResult.isNaN()) {
            println("loss error")
//            AList.forEach { println(it) }
            throw RuntimeException("loss error")
        }
        val DList = mutableListOf(loss.backward(AList.last(), output))
        layerList.zip(AList.dropLast(1)).asReversed().forEach { (layer, A) ->
            DList.add(layer.backward(DList.last(), A, learningRate = learningRate))
        }
        return lossResult
    }

    fun predict(input: D1Array<Double>): D1Array<Double> {
        val AList = mutableListOf(input.reshape(1, input.shape[0]))
        layerList.forEach { layer ->
            AList.add(layer.forward(AList.last()))
        }
        return AList.last().let { it.reshape(it.shape[0] * it.shape[1]) }
    }

    fun copy() = Model(*layerList.map { it.copy() }.toTypedArray(), loss = loss)
}