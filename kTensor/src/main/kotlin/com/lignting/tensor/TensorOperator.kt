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

///**
// * ## 整数Tensor与整数Tensor加法
// *
// * 整数Tensor与整数Tensor相加，支持广播机制。
// *
// * 返回结果shape与this相同，other会被广播到this的shape。
// *
// * @param other 另一个张量
// * @return 加法结果
// */
//operator fun Tensor<IntNumber>.plus(other: Tensor<IntNumber>): Tensor<IntNumber> = Tensor(
//    shape = this.shape,
//    data = this.data.zip(
//        if (other.shape.contentEquals(this.shape)) {
//            other
//        } else {
//            other.broadcast(this.shape)
//        }.data
//    ) { a, b -> (a.toInt() + b.toInt()).typeInt() }.toTypedArray(),
//)
//
///**
// * ## 整数Tensor与浮点Tensor加法
// *
// * 整数Tensor与浮点Tensor相加，支持广播机制。
// *
// * 返回结果shape与this相同，other会被广播到this的shape。
// *
// * @param other 另一个张量
// * @return 加法结果
// */
//operator fun Tensor<IntNumber>.plus(other: Tensor<DoubleNumber>): Tensor<DoubleNumber> = Tensor(
//    shape = this.shape,
//    data = this.data.zip(
//        if (other.shape.contentEquals(this.shape)) {
//            other
//        } else {
//            other.broadcast(this.shape)
//        }.data
//    ) { a, b -> (a.toInt() + b.toDouble()).typeDouble() }.toTypedArray(),
//)
//
///**
// * ## 浮点Tensor与浮点Tensor加法
// *
// * 浮点Tensor与浮点Tensor相加，支持广播机制。
// *
// * 返回结果shape与this相同，other会被广播到this的shape。
// *
// * @param other 另一个张量
// * @return 加法结果
// */
//operator fun Tensor<DoubleNumber>.plus(other: Tensor<DoubleNumber>): Tensor<DoubleNumber> = Tensor(
//    shape = this.shape,
//    data = this.data.zip(
//        if (other.shape.contentEquals(this.shape)) {
//            other
//        } else {
//            other.broadcast(this.shape)
//        }.data
//    ) { a, b -> (a.toDouble() + b.toDouble()).typeDouble() }.toTypedArray(),
//)
//
///**
// * ## 浮点Tensor与整数Tensor加法
// *
// * 浮点Tensor与整数Tensor相加，支持广播机制。
// *
// * 返回结果shape与this相同，other会被广播到this的shape。
// *
// * @param other 另一个张量
// * @return 加法结果
// */
//operator fun Tensor<DoubleNumber>.plus(other: Tensor<IntNumber>): Tensor<DoubleNumber> = Tensor(
//    shape = this.shape,
//    data = this.data.zip(
//        if (other.shape.contentEquals(this.shape)) {
//            other
//        } else {
//            other.broadcast(this.shape)
//        }.data
//    ) { a, b -> (a.toDouble() + b.toInt()).typeDouble() }.toTypedArray(),
//)