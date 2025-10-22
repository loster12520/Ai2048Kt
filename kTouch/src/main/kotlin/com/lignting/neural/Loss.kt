package com.lignting.neural

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.reduce
import org.jetbrains.kotlinx.multik.ndarray.operations.sum
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import kotlin.math.*

/**
 * ### 损失函数接口。
 *
 * 用于神经网络的损失计算与反向传播，所有损失函数需实现该接口。
 *
 * 主要方法：
 * - loss：计算损失值。
 * - backward：计算损失对输出的梯度。
 *
 * 常见实现类：
 * - Mse：均方误差，适合回归任务。
 * - Mae：平均绝对误差，抗异常值能力强。
 * - HuberLoss：结合MSE和MAE，兼顾鲁棒性与收敛速度。
 *
 * @param y 真实值，形状为(batchSize, outputSize)
 * @param yHat 预测值，形状为(batchSize, outputSize)
 * @return 损失值或梯度，具体见各实现
 */
interface Loss {
    /**
     * 损失计算
     * @param y 真实值，形状为(batchSize, outputSize)
     * @param yHat 预测值，形状为(batchSize, outputSize)
     * @return 损失值（Double）
     */
    fun loss(y: D2Array<Double>, yHat: D2Array<Double>): Double

    /**
     * 损失反向传播
     * @param y 真实值，形状为(batchSize, outputSize)
     * @param yHat 预测值，形状为(batchSize, outputSize)
     * @return 损失梯度（D2Array<Double>）
     */
    fun backward(y: D2Array<Double>, yHat: D2Array<Double>): D2Array<Double>
}

/**
 * ### 均方误差(MSE)损失。
 *
 * 适用于回归任务，对异常值敏感。
 *
 * 优点：收敛快，公式简单。
 * 缺点：对异常值敏感。
 *
 * 同类型对比：MAE更鲁棒但收敛慢，HuberLoss兼顾两者。
 *
 * 主要数学公式：
 * - $L = \frac{1}{n}\sum (y - \hat{y})^2$
 *
 * @param y 真实值，形状为(batchSize, outputSize)
 * @param yHat 预测值，形状为(batchSize, outputSize)
 * @return 损失值/梯度
 */
class Mse : Loss {
    override fun loss(y: D2Array<Double>, yHat: D2Array<Double>): Double {
        val n = y.shape[0] * y.shape[1]
        return (y - yHat).reduce { a, i ->
            a + (i.pow(2))
        } / n
    }

    override fun backward(y: D2Array<Double>, yHat: D2Array<Double>): D2Array<Double> {
        val n = y.shape[0] * y.shape[1]
        return (y - yHat).map { -it * 2 / n }
    }

}

/**
 * ### 平均绝对误差(MAE)损失。
 *
 * 适用于回归任务，对异常值鲁棒。
 *
 * 优点：抗异常值能力强。
 * 缺点：不可微，收敛慢。
 *
 * 同类型对比：MSE收敛快但对异常值敏感，HuberLoss兼顾两者。
 *
 * 主要数学公式：
 * - $L = \frac{1}{n}\sum |y - \hat{y}|$
 *
 * @param y 真实值，形状为(batchSize, outputSize)
 * @param yHat 预测值，形状为(batchSize, outputSize)
 * @return 损失值/梯度
 */
class Mae : Loss {
    override fun loss(y: D2Array<Double>, yHat: D2Array<Double>): Double {
        val n = y.shape[0] * y.shape[1]
        return (y - yHat).reduce { a, i ->
            a + (abs(i))
        } / n
    }

    override fun backward(y: D2Array<Double>, yHat: D2Array<Double>): D2Array<Double> {
        val n = y.shape[0] * y.shape[1]
        return (y - yHat).map { if (it > 0) -1.0 / n else if (it < 0) 1.0 / n else 0.0 }
    }
}

/**
 * ### Huber损失。
 *
 * 结合MSE和MAE，适合异常值场景。
 *
 * 优点：兼顾鲁棒性与收敛速度。
 * 缺点：需调节delta参数。
 *
 * 同类型对比：MSE收敛快但对异常值敏感，MAE鲁棒但收敛慢。
 *
 * 主要数学公式：
 * - $L = \begin{cases} 0.5 (y-\hat{y})^2, & |y-\hat{y}|\leq\delta \\ \delta(|y-\hat{y}|-0.5\delta), & |y-\hat{y}|>\delta \end{cases}$
 *
 * @param delta 阈值，控制MSE与MAE的切换，默认为1.0
 * @param y 真实值，形状为(batchSize, outputSize)
 * @param yHat 预测值，形状为(batchSize, outputSize)
 * @return 损失值/梯度
 */
class HuberLoss(private val delta: Double = 1.0) : Loss {
    override fun loss(y: D2Array<Double>, yHat: D2Array<Double>): Double {
        val n = y.shape[0] * y.shape[1]
        return (y - yHat).map { diff ->
            if (abs(diff) <= delta) {
                0.5 * diff.pow(2)
            } else {
                delta * (abs(diff) - 0.5 * delta)
            }
        }.sum() / n
    }

    override fun backward(y: D2Array<Double>, yHat: D2Array<Double>): D2Array<Double> {
        val n = y.shape[0] * y.shape[1]
        return (y - yHat).map { diff ->
            when {
                abs(diff) <= delta -> -diff / n
                diff > 0 -> -delta / n
                else -> delta / n
            }
        }
    }
}

class CrossEntropy(private val epsilon: Double = 1e-12) : Loss {
    override fun loss(y: D2Array<Double>, yHat: D2Array<Double>): Double {
        val batch = y.shape[0]
        var total = 0.0
        for (i in 0 until batch) {
            for (j in 0 until y.shape[1]) {
                val p = yHat[i, j].coerceIn(epsilon, 1.0 - epsilon)
                total += -y[i, j] * ln(p)
            }
        }
        return total / batch
    }

    override fun backward(y: D2Array<Double>, yHat: D2Array<Double>): D2Array<Double> {
        val batch = y.shape[0]
        val grads = List(batch) { MutableList(y.shape[1]) { 0.0 } }
        for (i in 0 until batch) {
            for (j in 0 until y.shape[1]) {
                val p = yHat[i, j].coerceIn(epsilon, 1.0 - epsilon)
                grads[i][j] = -y[i, j] / p / batch
            }
        }
        return mk.ndarray(grads.map { it.toList() })
    }
}
