package com.lignting.neural

import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.reduce
import org.jetbrains.kotlinx.multik.ndarray.operations.sum
import kotlin.math.abs
import kotlin.math.pow

/**
 * 损失函数接口。
 * 包含损失计算和反向传播。
 * @see [doc/neural.md](../../doc/neural.md)
 */
interface Loss {
    /**
     * 损失计算
     * @param y 真实值
     * @param yHat 预测值
     * @return 损失值
     * @see [doc/neural.md](../../doc/neural.md)
     */
    fun loss(y: D2Array<Double>, yHat: D2Array<Double>): Double
    /**
     * 损失反向传播
     * @param y 真实值
     * @param yHat 预测值
     * @return 损失梯度
     * @see [doc/neural.md](../../doc/neural.md)
     */
    fun backward(y: D2Array<Double>, yHat: D2Array<Double>): D2Array<Double>
}

/**
 * 均方误差(MSE)损失。
 * $L = \frac{1}{n}\sum (y - \hat{y})^2$
 * @see [doc/neural.md](../../doc/neural.md)
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
 * 平均绝对误差(MAE)损失。
 * $L = \frac{1}{n}\sum |y - \hat{y}|$
 * @see [doc/neural.md](../../doc/neural.md)
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
 * Huber损失。
 * 结合MSE和MAE，适合异常值场景。
 * @see [doc/neural.md](../../doc/neural.md)
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