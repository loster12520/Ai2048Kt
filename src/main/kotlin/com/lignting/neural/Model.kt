package com.lignting.neural

import org.jetbrains.kotlinx.multik.ndarray.data.D2Array

class Model(vararg layers: Layer, private val loss: Loss, private val learningRate: Double = 1.0) {
    private val layerList = layers.toList()
    fun fit(input: D2Array<Double>, output: D2Array<Double>) {
        val AList = mutableListOf(input)
        layerList.forEach { layer ->
            AList.add(layer.forward(AList.last()))
        }
        val lossResult = loss.loss(AList.last(), output)
        val DList = mutableListOf(loss.backward(AList.last(), output))
        layerList.zip(AList.drop(1)).asReversed().forEach { (layer, A) ->
            DList.add(layer.backward(DList.last(), A, learningRate = learningRate))
        }
    }

    fun predict(input: D2Array<Double>): D2Array<Double> {
        val AList = mutableListOf(input)
        layerList.forEach { layer ->
            AList.add(layer.forward(AList.last()))
        }
        return AList.last()
    }

    fun copy() = Model(*layerList.map { it.copy() }.toTypedArray(), loss = loss)
}