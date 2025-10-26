package com.lignting.number

/**
 * TODO 需要迁移至kTypes子模块
 *
 * > 本文件中只考虑支持IntNumber和FloatNumber两种类型，不考虑其他类型
 *
 * @author: lignting
 */

val error = IllegalArgumentException("Unsupported number type")

/**
 * ### IntNumber接口
 *
 * int基础数值类型接口，定义了基本的算术运算符重载
 *
 * @param value 整数值
 * @return 返回相加后的数值类型对象
 */
class IntNumber(private val value: Int) : IntNumberTypes {
    override fun plus(number: NumberTypes) = when (number) {
        is IntNumber -> {
            IntNumber(this.value + number.value)
        }

        is DoubleNumber -> {
            DoubleNumber(this.value.toDouble() + number.toDouble())
        }

        else -> {
            throw error
        }
    }

    override fun minus(number: NumberTypes) = when (number) {
        is IntNumber -> {
            IntNumber(this.value - number.value)
        }

        is DoubleNumber -> {
            DoubleNumber(this.value.toDouble() - number.toDouble())
        }

        else -> {
            throw error
        }
    }

    override fun times(number: NumberTypes) = when (number) {
        is IntNumber -> {
            IntNumber(this.value * number.value)
        }

        is DoubleNumber -> {
            DoubleNumber(this.value.toDouble() * number.toDouble())
        }

        else -> {
            throw error
        }
    }

    override fun div(number: NumberTypes) = when (number) {
        is IntNumber -> {
            IntNumber(this.value / number.value)
        }

        is DoubleNumber -> {
            DoubleNumber(this.value.toDouble() / number.toDouble())
        }

        else -> {
            throw error
        }
    }

    override fun toInt(): Int = value

    override fun toLong(): Long = value.toLong()

    override fun equals(other: Any?) = if (other is IntNumber) {
        this.value == other.value
    } else {
        false
    }

    override fun toString(): String {
        return value.toString()
    }

    override fun hashCode(): Int {
        return value.hashCode()
    }
}

/**
 * ### DoubleNumber接口
 *
 * double基础数值类型接口，定义了基本的算术运算符重载
 *
 * @param value 浮点数值
 * @return 返回相加后的数值类型对象
 */
class DoubleNumber(private val value: Double) : FloatNumberTypes {
    override fun plus(number: NumberTypes) = when (number) {
        is IntNumber -> {
            DoubleNumber(this.value + number.toInt().toFloat())
        }

        is DoubleNumber -> {
            DoubleNumber(this.value + number.value)
        }

        else -> {
            throw error
        }
    }

    override fun minus(number: NumberTypes) = when (number) {
        is IntNumber -> {
            DoubleNumber(this.value - number.toInt().toFloat())
        }

        is DoubleNumber -> {
            DoubleNumber(this.value - number.value)
        }

        else -> {
            throw error
        }
    }

    override fun times(number: NumberTypes) = when (number) {
        is IntNumber -> {
            DoubleNumber(this.value * number.toInt().toFloat())
        }

        is DoubleNumber -> {
            DoubleNumber(this.value * number.value)
        }

        else -> {
            throw error
        }
    }

    override fun div(number: NumberTypes) = when (number) {
        is IntNumber -> {
            DoubleNumber(this.value / number.toInt().toFloat())
        }

        is DoubleNumber -> {
            DoubleNumber(this.value / number.value)
        }

        else -> {
            throw error
        }
    }

    override fun toFloat(): Float = value.toFloat()

    override fun toDouble(): Double = value

    override fun equals(other: Any?) = if (other is DoubleNumber) {
        this.value == other.value
    } else {
        false
    }

    override fun toString(): String {
        return value.toString()
    }

    override fun hashCode(): Int {
        return value.hashCode()
    }
}

/**
 * ### Number类型转换扩展函数
 *
 * 将Kotlin的Number类型转换为自定义的NumberTypes类型
 *
 * 支持Byte, Short, Int, Long转换为IntNumber
 *
 * 支持Float, Double转换为DoubleNumber
 *
 * 示例：
 * ```kotlin
 * val intNum: NumberTypes = 5.type() // 转换为IntNumber
 * val doubleNum: NumberTypes = 5.5.type() // 转换为Double
 * ```
 *
 * @receiver Number Kotlin的Number类型
 * @return NumberTypes 自定义的NumberTypes类型
 * @throws IllegalArgumentException 如果传入的Number类型不受支持，则抛出异常
 */
fun Number.type() = when (this) {
    is Byte -> IntNumber(this.toInt())
    is Short -> IntNumber(this.toInt())
    is Int -> IntNumber(this)
    is Long -> IntNumber(this.toInt())
    is Float -> DoubleNumber(this.toDouble())
    is Double -> DoubleNumber(this)
    else -> throw error
}

/**
 * ### Number类型转换为IntNumber扩展函数
 *
 * 将Kotlin的Number类型转换为自定义的IntNumber类型
 *
 * 支持Byte, Short, Int, Long, Float, Double转换为IntNumber
 *
 * 示例：
 * ```kotlin
 * val intNum: IntNumber = 5.typeInt() // 转换为IntNumber
 * val intNumFromFloat: IntNumber = 5.5.typeInt() // 转换为IntNumber，结果为5
 * ```
 * @receiver Number Kotlin的Number类型
 * @return IntNumber 自定义的IntNumber类型
 * @throws IllegalArgumentException 如果传入的Number类型不受支持，则抛
 */
fun Number.typeInt() = when (this) {
    is Byte -> IntNumber(this.toInt())
    is Short -> IntNumber(this.toInt())
    is Int -> IntNumber(this)
    is Long -> IntNumber(this.toInt())
    is Float -> IntNumber(this.toInt())
    is Double -> IntNumber(this.toInt())
    else -> throw error
}

/**
 * ### Number类型转换为DoubleNumber扩展函数
 *
 * 将Kotlin的Number类型转换为自定义的DoubleNumber类型
 *
 * 支持Byte, Short, Int, Long, Float, Double转换为DoubleNumber
 *
 * 示例：
 * ```kotlin
 * val doubleNum: DoubleNumber = 5.typeDouble() // 转换为DoubleNumber，结果为5.0
 * val doubleNumFromFloat: DoubleNumber = 5.5.typeDouble() // 转换为DoubleNumber，结果为5.5
 * ```
 * @receiver Number Kotlin的Number类型
 * @return DoubleNumber 自定义的DoubleNumber类型
 * @throws IllegalArgumentException 如果传入的Number类型不受支持，则抛出异常
 */
fun Number.typeDouble() = when (this) {
    is Byte -> DoubleNumber(this.toDouble())
    is Short -> DoubleNumber(this.toDouble())
    is Int -> DoubleNumber(this.toDouble())
    is Long -> DoubleNumber(this.toDouble())
    is Float -> DoubleNumber(this.toDouble())
    is Double -> DoubleNumber(this)
    else -> throw error
}