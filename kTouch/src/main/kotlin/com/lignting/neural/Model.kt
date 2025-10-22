package com.lignting.neural

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.operations.toListD2

/**
 * ### 神经网络模型类。
 *
 * 该类用于构建和管理多层神经网络，集成了前向传播、反向传播、预测、评估等核心方法。
 * 支持自定义损失函数、优化器和学习率调度器。
 *
 * 同类型对比：
 * - 与传统ML库相比，支持更细粒度的自定义和扩展。
 * - 与Keras/PyTorch等高级框架相比，底层更透明但功能有限。
 *
 * 示例：
 * ```kotlin
 * val model = Model(Dense(4, 8), Relu(), Dense(8, 1))
 * ```
 *
 * 构造参数：
 * @param layers 神经网络层（可变参数，顺序即网络结构）
 * @param loss 损失函数，默认Mse
 * @param optimizer 优化器，默认GradientDescent
 * @param scheduler 学习率调度器，默认StepDecayScheduler(0.01)
 */
class Model(
    vararg layers: Layer,
    var loss: Loss = Mse(),
    var optimizer: Optimizer = GradientDescent(),
    var scheduler: Scheduler = StepDecayScheduler(0.01)
) {
    private val layerList = layers.toList()

    /**
     * ### 训练模型（单批次）
     *
     * 实现了标准的前向传播和反向传播流程。
     *
     * 示例：
     * ```kotlin
     * val loss = model.fit(input, output, epoch)
     * ```
     *
     * @param input 输入数据，形状为(batchSize, inputSize)
     * @param output 真实标签，形状为(batchSize, outputSize)
     * @param epoch 当前训练轮次
     * @return 当前批次的损失值
     */
    fun fit(input: D2Array<Double>, output: D2Array<Double>, epoch: Int): Double {
        val aList = mutableListOf(input)
        layerList.forEach { layer ->
            aList.add(layer.forward(aList.last()))
        }
        val lossResult = loss.loss(output, aList.last())
        if (lossResult.isNaN()) {
            println(
                "loss error $lossResult \n$" +
                    aList.map { it.toListD2().take(1) }.joinToString("\n---\n") { it.toString() }
            )
            throw RuntimeException("loss error")
        }
        val dList = mutableListOf(loss.backward(output, aList.last()))
        layerList.zip(aList.dropLast(1)).asReversed().forEach { (layer, A) ->
            dList.add(
                layer.backward(
                    A,
                    dList.last(),
                    optimizer.copy(),
                    scheduler,
                    epoch
                )
            )
        }
        return lossResult
    }

    /**
     * ### 批量训练
     *
     * 支持自定义批大小，自动分批训练并返回整体评估结果。
     *
     * 示例：
     * ```kotlin
     * val loss = model.fitWithBatchSize(input, output, epoch, batchSize = 32)
     * ```
     *
     * @param input 输入数据，形状为(batchSize, inputSize)
     * @param output 真实标签，形状为(batchSize, outputSize)
     * @param epoch 当前训练轮次
     * @param batchSize 批大小，默认100
     * @return 训练后整体评估损失
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
     * ### 单样本预测
     *
     * 示例：
     * ```kotlin
     * val yPred = model.predictOne(input)
     * ```
     *
     * @param input 单个样本输入，形状为(inputSize)
     * @return 单个样本预测结果，形状为(outputSize)
     */
    fun predictOne(input: D1Array<Double>): D1Array<Double> {
        val aList = mutableListOf(input.reshape(1, input.shape[0]))
        layerList.forEach { layer ->
            aList.add(layer.forward(aList.last()))
        }
        return aList.last().let { it.reshape(it.shape[0] * it.shape[1]) }
    }

    /**
     * ### 批量预测
     *
     * 示例：
     * ```kotlin
     * val yPred = model.predict(input)
     * ```
     *
     * @param input 批量输入数据，形状为(batchSize, inputSize)
     * @return 批量预测结果，形状为(batchSize, outputSize)
     */
    fun predict(input: D2Array<Double>): D2Array<Double> {
        val aList = mutableListOf(input)
        layerList.forEach { layer ->
            aList.add(layer.forward(aList.last()))
        }
        return aList.last()
    }

    /**
     * ### 评估模型
     *
     * 示例：
     * ```kotlin
     * val evalLoss = model.evaluate(input, output)
     * ```
     *
     * @param input 输入数据，形状为(batchSize, inputSize)
     * @param output 真实标签，形状为(batchSize, outputSize)
     * @return 损失值
     */
    fun evaluate(input: D2Array<Double>, output: D2Array<Double>) = loss.loss(output, predict(input))

    /**
     * ### 深拷贝模型
     *
     * 示例：
     * ```kotlin
     * val modelCopy = model.copy()
     * ```
     *
     * @return 新的Model实例
     */
    fun copy() = Model(*layerList.map { it.copy() }.toTypedArray(), loss = loss, optimizer = optimizer.copy())

    /**
     * ### 打印模型结构信息
     *
     * 示例：
     * ```kotlin
     * model.log()
     * ```
     *
     * @return 层信息字符串
     */
    fun log() = layerList.joinToString("\n") { it.info() }.also {
        println("model info: \n$it")
    }
}