package com.lignting.number

/**
 * 基础数值类型接口，定义了基本的算术运算符重载
 */
interface NumberTypes {
    /**
     * 加法运算符重载
     * @param number 另一个数值类型对象
     * @return 返回相加后的数值类型对象
     */
    operator fun plus(number: NumberTypes): NumberTypes

    /**
     * 减法运算符重载
     * @param number 另一个数值类型对象
     * @return 返回相减后的数值类型对象
     */
    operator fun minus(number: NumberTypes): NumberTypes

    /**
     * 乘法运算符重载
     * @param number 另一个数值类型对象
     * @return 返回相乘后的数值类型对象
     */
    operator fun times(number: NumberTypes): NumberTypes

    /**
     * 除法运算符重载
     * @param number 另一个数值类型对象
     * @return 返回相除后的数值类型对象
     */
    operator fun div(number: NumberTypes): NumberTypes
}

/**
 * 浮点数类型接口，继承自 NumberTypes，定义了转换为 Float 和 Double 的方法
 */
interface FloatNumberTypes : NumberTypes {
    /**
     * 转换为 Float 类型
     * @return 返回转换后的 Float 值
     */
    fun toFloat(): Float

    /**
     * 转换为 Double 类型
     * @return 返回转换后的 Double 值
     */
    fun toDouble(): Double
}

/**
 * 整数类型接口，继承自 NumberTypes，定义了转换为 Int 和 Long 的方法
 */
interface IntNumberTypes : NumberTypes {
    /**
     * 转换为 Int 类型
     * @return 返回转换后的 Int 值
     */
    fun toInt(): Int

    /**
     * 转换为 Long 类型
     * @return 返回转换后的 Long 值
     */
    fun toLong(): Long
}