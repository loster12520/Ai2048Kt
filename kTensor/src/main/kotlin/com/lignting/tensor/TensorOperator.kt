package com.lignting.tensor

import com.lignting.number.*

/**
 * ## 加法操作实现
 *
 * 加法操作的具体实现。
 *
 * @author Lightning
 */
object Plus : TwoTypeOperator {
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
    override val backwardIntWithInt = { _: Int, _: Int ->
        { grad: Int -> grad to grad }
    }
    override val backwardIntWithDouble = { _: Int, _: Double ->
        { grad: Double -> grad.toInt() to grad }
    }
    override val backwardDoubleWithInt = { _: Double, _: Int ->
        { grad: Double -> grad to grad.toInt() }
    }
    override val backwardDoubleWithDouble = { _: Double, _: Double ->
        { grad: Double -> grad to grad }
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
object Minus : TwoTypeOperator {
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
    override val backwardIntWithInt = { _: Int, _: Int ->
        { grad: Int -> -grad to -grad }
    }
    override val backwardIntWithDouble = { _: Int, _: Double ->
        { grad: Double -> -grad.toInt() to -grad }
    }
    override val backwardDoubleWithInt = { _: Double, _: Int ->
        { grad: Double -> -grad to -grad.toInt() }
    }
    override val backwardDoubleWithDouble = { _: Double, _: Double ->
        { grad: Double -> -grad to -grad }
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
object Times : TwoTypeOperator {
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
    override val backwardIntWithInt = { a: Int, b: Int ->
        { grad: Int -> grad * b to grad * b }
    }
    override val backwardIntWithDouble = { a: Int, b: Double ->
        { grad: Double -> (grad * b).toInt() to grad * a }
    }
    override val backwardDoubleWithInt = { a: Double, b: Int ->
        { grad: Double -> grad * b to (grad * a).toInt() }
    }
    override val backwardDoubleWithDouble = { a: Double, b: Double ->
        { grad: Double -> grad * b to grad * a }
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
object Div : TwoTypeOperator {
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
    override val backwardIntWithInt = { a: Int, b: Int ->
        { grad: Int -> grad / b to grad / b }
    }
    override val backwardIntWithDouble = { a: Int, b: Double ->
        { grad: Double -> (grad / b).toInt() to grad / a }
    }
    override val backwardDoubleWithInt = { a: Double, b: Int ->
        { grad: Double -> grad / b to (grad / a).toInt() }
    }
    override val backwardDoubleWithDouble = { a: Double, b: Double ->
        { grad: Double -> grad / b to grad / a }
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

/**
 * ## 取模操作实现
 *
 * 取模操作的具体实现。
 *
 * @author Lightning
 */
object Mod : TwoTypeOperator {
    override val intWithInt = { a: Int, b: Int ->
        a % b
    }
    override val intWithDouble = { a: Int, b: Double ->
        a % b
    }
    override val doubleWithInt = { a: Double, b: Int ->
        a % b
    }
    override val doubleWithDouble = { a: Double, b: Double ->
        a % b
    }
    override val backwardIntWithInt = { a: Int, b: Int ->
        { grad: Int ->
            // grad_a = grad, grad_b = -grad * (a / b)  (使用整数除法)
            grad to -(grad * (a / b))
        }
    }
    override val backwardIntWithDouble = { a: Int, b: Double ->
        { grad: Double ->
            // grad_a 作为 Int 返回，grad_b 为 Double，floor 使用 double 计算
            grad.toInt() to -grad * kotlin.math.floor(a.toDouble() / b)
        }
    }
    override val backwardDoubleWithInt = { a: Double, b: Int ->
        { grad: Double ->
            // grad_a 为 Double，grad_b 转为 Int
            grad to -( (grad * kotlin.math.floor(a / b)).toInt() )
        }
    }
    override val backwardDoubleWithDouble = { a: Double, b: Double ->
        { grad: Double ->
            // 全为 Double 类型
            grad to -grad * kotlin.math.floor(a / b)
        }
    }
}

/**
 * ## Tensor整数标量取模
 *
 * Tensor与整数标量取模。
 *
 * @param other 整数标量
 * @return 取模结果
 */
inline operator fun <reified T : NumberTypes> Tensor<T>.rem(other: Int) =
    tensorWith(other, Mod)

/**
 * ## Tensor浮点标量取模
 *
 * Tensor与浮点标量取模。
 *
 * @param other 浮点标量
 * @return 取模结果
 */
inline operator fun <reified T : NumberTypes> Tensor<T>.rem(other: Double) =
    tensorWith(other, Mod)

/**
 * ## Tensor与Tensor取模
 *
 * Tensor与Tensor按元素取模，支持广播机制。
 *
 * @param other 另一个Tensor
 * @return 取模结果
 */
inline operator fun <reified T : NumberTypes, reified F : NumberTypes> Tensor<T>.rem(other: Tensor<F>): Tensor<NumberTypes> =
    tensorWith(other, Mod)