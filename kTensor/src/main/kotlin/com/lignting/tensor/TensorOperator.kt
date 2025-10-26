package com.lignting.tensor

import com.lignting.number.*
import kotlin.collections.map

/**
 * ## Tensor整数标量加法
 *
 * Tensor与整数标量相加。
 *
 * @param other 整数标量
 * @return 加法结果
 */
inline operator fun <reified T : NumberTypes> Tensor<T>.plus(other: Int) =
    when (T::class) {
        IntNumber::class -> Tensor(
            shape = this.shape,
            data = this.data.map { (it as IntNumber).toInt() + other }.map { it.typeInt() }.toTypedArray(),
        )

        DoubleNumber::class -> Tensor(
            shape = this.shape,
            data = this.data.map { (it as DoubleNumber).toDouble() + other }.map { it.typeDouble() }.toTypedArray(),
        )

        else -> throw IllegalArgumentException("Unsupported type")
    }

/**
 * ## Tensor整数标量与浮点标量加法
 *
 * Tensor与浮点标量相加。
 *
 * @param other 浮点标量
 * @return 加法结果
 */
inline operator fun <reified T : NumberTypes> Tensor<T>.plus(other: Double) =
    when (T::class) {
        IntNumber::class -> Tensor(
            shape = this.shape,
            data = this.data.map { (it as IntNumber).toInt() + other }.map { it.typeDouble() }.toTypedArray(),
        )

        DoubleNumber::class -> Tensor(
            shape = this.shape,
            data = this.data.map { (it as DoubleNumber).toDouble() + other }.map { it.typeDouble() }.toTypedArray(),
        )

        else -> throw IllegalArgumentException("Unsupported type")
    }

/**
 * ## Tensor与Tensor加法
 *
 * Tensor与Tensor按元素相加，支持广播机制。
 *
 * @param other 另一个Tensor
 * @return 加法结果
 */
inline operator fun <reified T : NumberTypes, reified F : NumberTypes> Tensor<T>.plus(other: Tensor<F>): Tensor<NumberTypes> =
    when {
        T::class == IntNumber::class && F::class == IntNumber::class -> Tensor(
            shape = this.shape,
            data = this.data.zip(
                if (other.shape.contentEquals(this.shape)) {
                    other
                } else {
                    other.broadcast(this.shape)
                }.data
            ) { a, b -> ((a as IntNumber).toInt() + (b as IntNumber).toInt()).typeInt() }.toTypedArray(),
        )

        T::class == IntNumber::class && F::class == DoubleNumber::class -> Tensor(
            shape = this.shape,
            data = this.data.zip(
                if (other.shape.contentEquals(this.shape)) {
                    other
                } else {
                    other.broadcast(this.shape)
                }.data
            ) { a, b -> ((a as IntNumber).toInt() + (b as DoubleNumber).toDouble()).typeDouble() }.toTypedArray(),
        )

        T::class == DoubleNumber::class && F::class == DoubleNumber::class -> Tensor(
            shape = this.shape,
            data = this.data.zip(
                if (other.shape.contentEquals(this.shape)) {
                    other
                } else {
                    other.broadcast(this.shape)
                }.data
            ) { a, b -> ((a as DoubleNumber).toDouble() + (b as DoubleNumber).toDouble()).typeDouble() }.toTypedArray(),
        )

        T::class == DoubleNumber::class && F::class == IntNumber::class -> Tensor(
            shape = this.shape,
            data = this.data.zip(
                if (other.shape.contentEquals(this.shape)) {
                    other
                } else {
                    other.broadcast(this.shape)
                }.data
            ) { a, b -> ((a as DoubleNumber).toDouble() + (b as IntNumber).toInt()).typeDouble() }.toTypedArray(),
        )

        else -> throw IllegalArgumentException("Unsupported type")
    }

