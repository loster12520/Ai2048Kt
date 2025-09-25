package com.lignting.neural

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.operations.toListD2

/**
 * 神经网络模型类，包含多层结构、损失函数、优化器和学习率调度器。
 * 支持训练、预测、评估和模型复制。
 */
class Model(
    vararg layers: Layer,
    private val loss: Loss = Mse(),
    private val optimizer: Optimizer = GradientDescent(),
    private val scheduler: Scheduler = StepDecayScheduler(0.01)
) {
    private val layerList = layers.toList()

    /**
     * 训练模型，单次迭代。
     * @param input 输入数据
     * @param output 真实标签
     * @param epoch 当前轮次
     * @return 损失值
     */
    fun fit(input: D2Array<Double>, output: D2Array<Double>, epoch: Int): Double {
        val aList = mutableListOf(input)
        layerList.forEach { layer ->
            aList.add(layer.forward(aList.last()))
        }
        val lossResult = loss.loss(output, aList.last())
        if (lossResult.isNaN()) {
            println(
                "loss error $lossResult \n${
                    aList.map { it.toListD2().take(1) }.joinToString("\n---\n") { it.toString() }
                }"
            )
            throw RuntimeException("loss error")
        }
        val dList = mutableListOf(loss.backward(output, aList.last()))
        layerList.zip(aList.dropLast(1)).asReversed().forEach { (layer, A) ->
            dList.add(
                layer.backward(
                    dList.last(),
                    A,
                    optimizer.copy(),
                    scheduler,
                    epoch
                )
            )
        }
        return lossResult
    }

    /**
     * 批量训练模型。
     * @param input 输入数据
     * @param output 真实标签
     * @param epoch 当前轮次
     * @param batchSize 批大小
     * @return 损失值
     */
    fun fitWithBatchSize(input: D2Array<Double>, output: D2Array<Double>, epoch: Int, batchSize: Int = 100): Double {
        input.toListD2().zip(output.toListD2()).shuffled().chunked(batchSize).forEach {
            val input = mk.ndarray(it.map { it.first })
            val output = mk.ndarray(it.map { it.second })
            fit(input, output, epoch)
        }
        return evaluate(input, output)
    }

    /**
     * 单样本预测。
     * @param input 输入数据
     * @return 预测结果
     */
    fun predictOne(input: D1Array<Double>): D1Array<Double> {
        val aList = mutableListOf(input.reshape(1, input.shape[0]))
        layerList.forEach { layer ->
            aList.add(layer.forward(aList.last()))
        }
        return aList.last().let { it.reshape(it.shape[0] * it.shape[1]) }
    }

    /**
     * 批量预测。
     * @param input 输入数据
     * @return 预测结果
     */
    fun predict(input: D2Array<Double>): D2Array<Double> {
        val aList = mutableListOf(input)
        layerList.forEach { layer ->
            aList.add(layer.forward(aList.last()))
        }
        return aList.last()
    }

    /**
     * 评估模型。
     * @param input 输入数据
     * @param output 真实标签
     * @return 损失值
     */
    fun evaluate(input: D2Array<Double>, output: D2Array<Double>) = loss.loss(output, predict(input))

    /**
     * 复制模型。
     */
    fun copy() = Model(*layerList.map { it.copy() }.toTypedArray(), loss = loss, optimizer = optimizer.copy())

    /**
     * 打印模型结构信息。
     */
    fun log() = layerList.joinToString("\n") { it.info() }.also {
        println("model info: \n$it")
    }
}