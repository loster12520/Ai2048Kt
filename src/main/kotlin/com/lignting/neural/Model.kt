package com.lignting.neural

import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import kotlin.math.min

class Model(vararg layers: Layer, private val loss: Loss = Mse(), private val learningRate: Double = 0.01,private val optimizer: Optimizer = GradientDescent()) {
    private val layerList = layers.toList()
    fun fit(
        input: D2Array<Double>,
        output: D2Array<Double>,
        learningRateReductionRate: Int = 0,
        learningRateDeclareDistance: Double = 0.00001
    ): Double {
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
            DList.add(
                layer.backward(
                    DList.last(),
                    A,
                    learningRate = learningRate - min(
                        learningRateReductionRate,
                        (learningRate / learningRateDeclareDistance).toInt() - 1
                    ) * learningRateDeclareDistance
                )
            )
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