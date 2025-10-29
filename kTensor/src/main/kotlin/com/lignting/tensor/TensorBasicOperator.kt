package com.lignting.tensor

import com.lignting.number.*

/**
 * ## 双参数操作接口
 *
 * 定义了int和double互相的操作，用来使用下面的`tensorWith`函数实现各种算术运算。
 *
 * @author Lightning
 */
interface TwoTypeOperator {
    val intWithInt: (Int, Int) -> Int
    val intWithDouble: (Int, Double) -> Double
    val doubleWithInt: (Double, Int) -> Double
    val doubleWithDouble: (Double, Double) -> Double
    val backwardIntWithInt: (Int, Int) -> (Int) -> Pair<Int, Int>
    val backwardIntWithDouble: (Int, Double) -> (Double) -> Pair<Int, Double>
    val backwardDoubleWithInt: (Double, Int) -> (Double) -> Pair<Double, Int>
    val backwardDoubleWithDouble: (Double, Double) -> (Double) -> Pair<Double, Double>
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
inline fun <reified T : NumberTypes> Tensor<T>.tensorWith(other: Int, typeOperator: TwoTypeOperator) =
    when (T::class) {
        IntNumber::class -> Tensor(
            shape = this.shape,
            data = this.data.map { (it as IntNumber).toInt().let { typeOperator.intWithInt(it, other) } }
                .map { it.typeInt() }.toTypedArray(),
            backwardFunction = {
                grad = data
                    .map { (it as IntNumber).toInt() }
                    .toIntArray().let { tensorIntOf(*it) } as Tensor<T>?
            },
            updateList = updateList.map { it.toIntTensor() } + this.toIntTensor()
        )
        
        DoubleNumber::class -> Tensor(
            shape = this.shape,
            data = this.data.map { (it as DoubleNumber).toDouble().let { typeOperator.doubleWithInt(it, other) } }
                .map { it.typeDouble() }.toTypedArray(),
            backwardFunction = {
                typeOperator.backwardIntWithInt
            },
            updateList = updateList.map { it.toDoubleTensor() } + this.toDoubleTensor()
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
inline fun <reified T : NumberTypes> Tensor<T>.tensorWith(other: Double, typeOperator: TwoTypeOperator) =
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
    typeOperator: TwoTypeOperator
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
 * ## 单参数操作接口
 *
 * 定义了对单个Int或Double类型参数进行操作的接口。
 *
 * @author Lightning
 */
interface OneTypeOperator {
    val int: (Int) -> Int
    val double: (Double) -> Double
}

/**
 * ## Tensor与标量操作
 *
 * Tensor与标量进行指定操作，返回新的Tensor。
 *
 * @param typeOperator 类型操作接口
 * @return 操作结果Tensor
 */
inline fun <reified T : NumberTypes> Tensor<T>.tensorWith(typeOperator: OneTypeOperator) =
    when (T::class) {
        IntNumber::class -> Tensor(
            shape = this.shape,
            data = this.data.map { (it as IntNumber).toInt().let { typeOperator.int(it) } }
                .map { it.typeInt() }.toTypedArray(),
        )
        
        DoubleNumber::class -> Tensor(
            shape = this.shape,
            data = this.data.map { (it as DoubleNumber).toDouble().let { typeOperator.double(it) } }
                .map { it.typeDouble() }.toTypedArray(),
        )
        
        else -> throw IllegalArgumentException("Unsupported type")
    }

/**
 * ## Tensor与Tensor操作
 *
 * Tensor与Tensor按元素进行指定操作，支持广播机制，返回新的Tensor。
 *
 * @param typeOperator 类型操作接口
 * @return 操作结果Tensor
 */
inline fun <reified T : NumberTypes, reified F : NumberTypes> Tensor<T>.tensorWith(
    typeOperator: TwoTypeOperator
): Tensor<NumberTypes> =
    when {
        T::class == IntNumber::class && F::class == IntNumber::class -> Tensor(
            shape = shape,
            data = data.zip(
                data
            ) { a, b -> typeOperator.intWithInt((a as IntNumber).toInt(), (b as IntNumber).toInt()).typeInt() }
                .toTypedArray(),
        )
        
        T::class == IntNumber::class && F::class == DoubleNumber::class -> Tensor(
            shape = shape,
            data = data.zip(
                data
            ) { a, b ->
                typeOperator.intWithDouble((a as IntNumber).toInt(), (b as DoubleNumber).toDouble()).typeDouble()
            }.toTypedArray(),
        )
        
        T::class == DoubleNumber::class && F::class == DoubleNumber::class -> Tensor(
            shape = this.shape,
            data = data.zip(
                data
            ) { a, b ->
                typeOperator.doubleWithDouble((a as DoubleNumber).toDouble(), (b as DoubleNumber).toDouble())
                    .typeDouble()
            }.toTypedArray(),
        )
        
        T::class == DoubleNumber::class && F::class == IntNumber::class -> Tensor(
            shape = this.shape,
            data = data.zip(
                data
            ) { a, b ->
                typeOperator.doubleWithInt((a as DoubleNumber).toDouble(), (b as IntNumber).toInt()).typeDouble()
            }.toTypedArray(),
        )
        
        else -> throw IllegalArgumentException("Unsupported type")
    }