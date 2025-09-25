package com.lignting.test

import com.lignting.neural.*
import org.jetbrains.kotlinx.dataframe.api.dataFrameOf
import org.jetbrains.kotlinx.kandy.dsl.plot
import org.jetbrains.kotlinx.kandy.letsplot.export.save
import org.jetbrains.kotlinx.kandy.letsplot.layers.line
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import kotlin.random.Random

fun test(times: Int): Double {
    val args = List(times) { Random.nextDouble() }

    // 测试函数
    fun function(para: List<Double>) = para.mapIndexed { index, value -> args[index] * value }.sum()

    // 获取数据函数，返回值为输入 to 输出
    fun getData(count: Int) =
        (0..count).map { List(times) { Random.nextDouble() * 10 } }
            .map { it to function(it) }.let {
                it.map { it.first } to it.map { it.second }
            }.let {
                mk.ndarray(it.first.map { it }) to mk.ndarray(it.second.map { listOf(it) })
            }

    // 构建模型
    val model = Model(
        Dense(times, 1),
        loss = Mse(),
        optimizer = GradientDescent(),
        scheduler = ExponentialScheduler(1e-3, dropRate = 1 - 1e-4)
    )
    // 构建数据
    val (trainX, trainY) = getData(1000)
    val (testX, testY) = getData(200)
    // 训练模型
    val (lossList, evaluateList) = (0..100).map {
        val loss = model.fit(trainX, trainY, 1)
        val evaluate = model.evaluate(testX, testY)
        loss to evaluate
    }.let {
        it.map { it.first } to it.map { it.second }
    }

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
    plot.save("loss_plot.png")

    // 测试预测
    val result = testX.let {
        model.predict(it).toList()
    }.zip(testY.toList()).map { (pred, real) ->
        pred to real
    }
    val rate = rate(result).let {
        (it * 10000).toInt() / 100.0
    }
    return rate
}

fun rate(result: List<Pair<Double, Double>>): Double {
    val yTrue = result.map { it.second }
    val yPred = result.map { it.first }
    val yMean = yTrue.average()
    val ssRes = yTrue.zip(yPred).sumOf { (real, pred) -> (real - pred) * (real - pred) }
    val ssTot = yTrue.sumOf { real -> (real - yMean) * (real - yMean) }
    val r2 = 1 - ssRes / ssTot
    return r2
}

fun main() {
    (2..10).map { times ->
        (1..100).map {
            test(times)
        }.let {
            (it.sum() / it.size * 100).toInt() / 100.0
        }.also {
            println("参数数量${times}，百次平均正确率: ${it}%")
        }
    }
}