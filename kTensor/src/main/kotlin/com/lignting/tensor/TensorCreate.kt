package com.lignting.tensor

import com.lignting.number.DoubleNumber
import com.lignting.number.IntNumber
import com.lignting.number.typeDouble
import com.lignting.number.typeInt

/**
 * TODO 需要迁移至kTypes子模块
 *
 * > 本文件中只考虑支持IntNumber和FloatNumber两种类型，不考虑其他类型
 *
 * @author: lignting
 */

/**
 * ### 零int tensor创建函数
 *
 * 创建一个指定形状的全零整数tensor
 * @param shape tensor形状
 * @return 返回一个全零的整数tensor
 */
fun zeroIntTensor(shape: IntArray): Tensor<IntNumber> {
    val size = shape.reduce { acc, i -> acc * i }
    val data = Array(size) { 0.typeInt() }
    return Tensor(shape, data)
}

/**
 * ### 零int tensor创建函数（快捷调用）
 *
 * 创建一个指定形状的全零整数tensor
 * @param shape tensor的形状，使用变长参数传递
 * @return 返回一个全零的整数tensor
 */
fun zeroI(vararg shape: Int): Tensor<IntNumber> {
    return zeroIntTensor(shape)
}

/**
 * ### 一int tensor创建函数
 *
 * 创建一个指定形状的全一整数tensor
 * @param shape tensor的形状
 * @return 返回一个全一的整数tensor
 */
fun onesIntTensor(shape: IntArray): Tensor<IntNumber> {
    val size = shape.reduce { acc, i -> acc * i }
    val data = Array(size) { 1.typeInt() }
    return Tensor(shape, data)
}

/**
 * ### 一int tensor创建函数（快捷调用）
 *
 * 创建一个指定形状的全一整数tensor
 * @param shape tensor的形状，使用变长参数传递
 * @return 返回一个全一的整数tensor
 */
fun onesI(vararg shape: Int): Tensor<IntNumber> {
    return onesIntTensor(shape)
}

/**
 * ### 全值int tensor创建函数
 *
 * 创建一个指定形状和填充值的整数tensor
 * @param shape tensor的形状
 * @param value 填充值
 * @return 返回一个指定填充值的整数tensor
 */
fun fullIntTensor(shape: IntArray, value: IntNumber): Tensor<IntNumber> {
    val size = shape.reduce { acc, i -> acc * i }
    val data = Array(size) { value }
    return Tensor(shape, data)
}

/**
 * ### 全值int tensor创建函数（快捷调用）
 *
 * 创建一个指定形状和填充值的整数tensor
 * @param value 填充值
 * @param shape tensor的形状，使用变长参数传递
 * @return 返回一个指定填充值的整数tensor
 */
fun fullI(value: IntNumber, vararg shape: Int): Tensor<IntNumber> {
    return fullIntTensor(shape, value)
}

/**
 * ### 零double tensor创建函数
 *
 * 创建一个指定形状的全零浮点数tensor
 * @param shape tensor形状
 * @return 返回一个全零的浮点数tensor
 */
fun zeroDoubleTensor(shape: IntArray): Tensor<DoubleNumber> {
    val size = shape.reduce { acc, i -> acc * i }
    val data = Array(size) { 0.0.typeDouble() }
    return Tensor(shape, data)
}

/**
 * ### 零double tensor创建函数（快捷调用）
 *
 * 创建一个指定形状的全零浮点数tensor
 * @param shape tensor的形状，使用变长参数传递
 * @return 返回一个全零的浮点数tensor
 */
fun zeroD(vararg shape: Int): Tensor<DoubleNumber> {
    return zeroDoubleTensor(shape)
}

/**
 * ### 一double tensor创建函数
 *
 * 创建一个指定形状的全一浮点数tensor
 * @param shape tensor的形状
 * @return 返回一个全一的浮点数tensor
 */
fun onesDoubleTensor(shape: IntArray): Tensor<DoubleNumber> {
    val size = shape.reduce { acc, i -> acc * i }
    val data = Array(size) { 1.0.typeDouble() }
    return Tensor(shape, data)
}

/**
 * ### 一double tensor创建函数（快捷调用）
 *
 * 创建一个指定形状的全一浮点数tensor
 * @param shape tensor的形状，使用变长参数传递
 * @return 返回一个全一的浮点数tensor
 */
fun onesD(vararg shape: Int): Tensor<DoubleNumber> {
    return onesDoubleTensor(shape)
}

/**
 * ### 全值double tensor创建函数
 *
 * 创建一个指定形状和填充值的浮点数tensor
 * @param shape tensor的形状
 * @param value 填充值
 * @return 返回一个指定填充值的浮点数tensor
 */
fun fullDoubleTensor(shape: IntArray, value: DoubleNumber): Tensor<DoubleNumber> {
    val size = shape.reduce { acc, i -> acc * i }
    val data = Array(size) { value }
    return Tensor(shape, data)
}

/**
 * ### 全值double tensor创建函数（快捷调用）
 *
 * 创建一个指定形状和填充值的浮点数tensor
 * @param value 填充值
 * @param shape tensor的形状，使用变长参数传递
 * @return 返回一个指定填充值的浮点数tensor
 */
fun fullD(value: DoubleNumber, vararg shape: Int): Tensor<DoubleNumber> {
    return fullDoubleTensor(shape, value)
}