package com.lignting.tensor

import com.lignting.number.*
import kotlin.collections.map

/**
 * ## 类型操作接口
 *
 * 定义了int和double互相的操作，用来使用下面的`tensorWith`函数实现各种算术运算。
 *
 * @author Lightning
 */
interface TypeOperator {
    val intWithInt: (Int, Int) -> Int
    val intWithDouble: (Int, Double) -> Double
    val doubleWithInt: (Double, Int) -> Double
    val doubleWithDouble: (Double, Double) -> Double
}

/**
 * ## Tensor与标量操作
 *
 * Tensor与标量进行指定操作，返回新的Tensor。
 *
 * @param other 标量
 * @param typeOperator 类型操作接口
 * @return 操作结果Tensor
 */
inline fun <reified T : NumberTypes> Tensor<T>.tensorWith(other: Int, typeOperator: TypeOperator) =
    when (T::class) {
        IntNumber::class -> Tensor(
            shape = this.shape,
            data = this.data.map { (it as IntNumber).toInt().let { typeOperator.intWithInt(it, other) } }
                .map { it.typeInt() }.toTypedArray(),
        )

        DoubleNumber::class -> Tensor(
            shape = this.shape,
            data = this.data.map { (it as DoubleNumber).toDouble().let { typeOperator.doubleWithInt(it, other) } }
                .map { it.typeDouble() }.toTypedArray(),
        )

        else -> throw IllegalArgumentException("Unsupported type")
    }

/**
 * ## Tensor与标量操作
 *
 * Tensor与标量进行指定操作，返回新的Tensor。
 *
 * @param other 标量
 * @param typeOperator 类型操作接口
 * @return 操作结果Tensor
 */
inline fun <reified T : NumberTypes> Tensor<T>.tensorWith(other: Double, typeOperator: TypeOperator) =
    when (T::class) {
        IntNumber::class -> Tensor(
            shape = this.shape,
            data = this.data.map { (it as IntNumber).toInt().let { typeOperator.intWithDouble(it, other) } }
                .map { it.typeInt() }.toTypedArray(),
        )

        DoubleNumber::class -> Tensor(
            shape = this.shape,
            data = this.data.map { (it as DoubleNumber).toDouble().let { typeOperator.doubleWithDouble(it, other) } }
                .map { it.typeDouble() }.toTypedArray(),
        )

        else -> throw IllegalArgumentException("Unsupported type")
    }

/**
 * ## Tensor与Tensor操作
 *
 * Tensor与Tensor按元素进行指定操作，支持广播机制，返回新的Tensor。
 *
 * @param other 另一个Tensor
 * @param typeOperator 类型操作接口
 * @return 操作结果Tensor
 */
inline fun <reified T : NumberTypes, reified F : NumberTypes> Tensor<T>.tensorWith(
    other: Tensor<F>,
    typeOperator: TypeOperator
): Tensor<NumberTypes> =
    when {
        T::class == IntNumber::class && F::class == IntNumber::class -> Tensor(
            shape = this.shape,
            data = this.data.zip(
                if (other.shape.contentEquals(this.shape)) {
                    other
                } else {
                    other.broadcast(this.shape)
                }.data
            ) { a, b -> typeOperator.intWithInt((a as IntNumber).toInt(), (b as IntNumber).toInt()).typeInt() }
                .toTypedArray(),
        )

        T::class == IntNumber::class && F::class == DoubleNumber::class -> Tensor(
            shape = this.shape,
            data = this.data.zip(
                if (other.shape.contentEquals(this.shape)) {
                    other
                } else {
                    other.broadcast(this.shape)
                }.data
            ) { a, b ->
                typeOperator.intWithDouble((a as IntNumber).toInt(), (b as DoubleNumber).toDouble()).typeDouble()
            }.toTypedArray(),
        )

        T::class == DoubleNumber::class && F::class == DoubleNumber::class -> Tensor(
            shape = this.shape,
            data = this.data.zip(
                if (other.shape.contentEquals(this.shape)) {
                    other
                } else {
                    other.broadcast(this.shape)
                }.data
            ) { a, b ->
                typeOperator.doubleWithDouble((a as DoubleNumber).toDouble(), (b as DoubleNumber).toDouble())
                    .typeDouble()
            }.toTypedArray(),
        )

        T::class == DoubleNumber::class && F::class == IntNumber::class -> Tensor(
            shape = this.shape,
            data = this.data.zip(
                if (other.shape.contentEquals(this.shape)) {
                    other
                } else {
                    other.broadcast(this.shape)
                }.data
            ) { a, b ->
                typeOperator.doubleWithInt((a as DoubleNumber).toDouble(), (b as IntNumber).toInt()).typeDouble()
            }.toTypedArray(),
        )

        else -> throw IllegalArgumentException("Unsupported type")
    }

/**
 * ## 加法操作实现
 *
 * 加法操作的具体实现。
 *
 * @author Lightning
 */
object Plus : TypeOperator {
    override val intWithInt = { a: Int, b: Int ->
        a + b
    }
    override val intWithDouble = { a: Int, b: Double ->
        a + b
    }
    override val doubleWithInt = { a: Double, b: Int ->
        a + b
    }
    override val doubleWithDouble = { a: Double, b: Double ->
        a + b
    }
}

/**
 * ## Tensor整数标量加法
 *
 * Tensor与整数标量相加。
 *
 * @param other 整数标量
 * @return 加法结果
 */
inline operator fun <reified T : NumberTypes> Tensor<T>.plus(other: Int) =
    tensorWith(other, Plus)

/**
 * ## Tensor整数标量与浮点标量加法
 *
 * Tensor与浮点标量相加。
 *
 * @param other 浮点标量
 * @return 加法结果
 */
inline operator fun <reified T : NumberTypes> Tensor<T>.plus(other: Double) =
    tensorWith(other, Plus)

/**
 * ## Tensor与Tensor加法
 *
 * Tensor与Tensor按元素相加，支持广播机制。
 *
 * @param other 另一个Tensor
 * @return 加法结果
 */
inline operator fun <reified T : NumberTypes, reified F : NumberTypes> Tensor<T>.plus(other: Tensor<F>): Tensor<NumberTypes> =
    tensorWith(other, Plus)

/**
 * ## 减法操作实现
 *
 * 减法操作的具体实现。
 *
 * @author Lightning
 */
object Minus : TypeOperator {
    override val intWithInt = { a: Int, b: Int ->
        a - b
    }
    override val intWithDouble = { a: Int, b: Double ->
        a - b
    }
    override val doubleWithInt = { a: Double, b: Int ->
        a - b
    }
    override val doubleWithDouble = { a: Double, b: Double ->
        a - b
    }
}

/**
 * ## Tensor整数标量减法
 *
 * Tensor与整数标量相减。
 *
 * @param other 整数标量
 * @return 减法结果
 */
inline operator fun <reified T : NumberTypes> Tensor<T>.minus(other: Int) =
    tensorWith(other, Minus)

/**
 * ## Tensor浮点标量减法
 *
 * Tensor与浮点标量相减。
 *
 * @param other 浮点标量
 * @return 减法结果
 */
inline operator fun <reified T : NumberTypes> Tensor<T>.minus(other: Double) =
    tensorWith(other, Minus)

/**
 * ## Tensor与Tensor减法
 *
 * Tensor与Tensor按元素相减，支持广播机制。
 *
 * @param other 另一个Tensor
 * @return 减法结果
 */
inline operator fun <reified T : NumberTypes, reified F : NumberTypes> Tensor<T>.minus(other: Tensor<F>): Tensor<NumberTypes> =
    tensorWith(other, Minus)

/**
 * ## 乘法操作实现
 *
 * 乘法操作的具体实现。
 *
 * @author Lightning
 */
object Times : TypeOperator {
    override val intWithInt = { a: Int, b: Int ->
        a * b
    }
    override val intWithDouble = { a: Int, b: Double ->
        a * b
    }
    override val doubleWithInt = { a: Double, b: Int ->
        a * b
    }
    override val doubleWithDouble = { a: Double, b: Double ->
        a * b
    }
}

/**
 * ## Tensor整数标量乘法
 *
 * Tensor与整数标量相乘。
 *
 * @param other 整数标量
 * @return 乘法结果
 */
inline operator fun <reified T : NumberTypes> Tensor<T>.times(other: Int) =
    tensorWith(other, Times)

/**
 * ## Tensor浮点标量乘法
 *
 * Tensor与浮点标量相乘。
 *
 * @param other 浮点标量
 * @return 乘法结果
 */
inline operator fun <reified T : NumberTypes> Tensor<T>.times(other: Double) =
    tensorWith(other, Times)

/**
 * ## Tensor与Tensor乘法
 *
 * Tensor与Tensor按元素相乘，支持广播机制。
 *
 * @param other 另一个Tensor
 * @return 乘法结果
 */
inline operator fun <reified T : NumberTypes, reified F : NumberTypes> Tensor<T>.times(other: Tensor<F>): Tensor<NumberTypes> =
    tensorWith(other, Times)

/**
 * ## 除法操作实现
 *
 * 除法操作的具体实现。
 *
 * @author Lightning
 */
object Div : TypeOperator {
    override val intWithInt = { a: Int, b: Int ->
        a / b
    }
    override val intWithDouble = { a: Int, b: Double ->
        a / b
    }
    override val doubleWithInt = { a: Double, b: Int ->
        a / b
    }
    override val doubleWithDouble = { a: Double, b: Double ->
        a / b
    }
}

/**
 * ## Tensor整数标量除法
 *
 * Tensor与整数标量相除。
 *
 * @param other 整数标量
 * @return 除法结果
 */
inline operator fun <reified T : NumberTypes> Tensor<T>.div(other: Int) =
    tensorWith(other, Div)

/**
 * ## Tensor浮点标量除法
 *
 * Tensor与浮点标量相除。
 *
 * @param other 浮点标量
 * @return 除法结果
 */
inline operator fun <reified T : NumberTypes> Tensor<T>.div(other: Double) =
    tensorWith(other, Div)

/**
 * ## Tensor与Tensor除法
 *
 * Tensor与Tensor按元素相除，支持广播机制。
 *
 * @param other 另一个Tensor
 * @return 除法结果
 */
inline operator fun <reified T : NumberTypes, reified F : NumberTypes> Tensor<T>.div(other: Tensor<F>): Tensor<NumberTypes> =
    tensorWith(other, Div)