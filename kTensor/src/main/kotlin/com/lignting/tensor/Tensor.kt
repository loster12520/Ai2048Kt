package com.lignting.tensor

import com.lignting.number.DoubleNumber
import com.lignting.number.IntNumber
import com.lignting.number.NumberTypes

/**
 * ## n维向量
 *
 * 这是一个n维向量实现，支持基本的形状、数据存储和反向传播功能。
 *
 * 注意，任意的`Tensor`实例均不可变，即其`shape`和`data`属性在实例化后均不可更改。
 *
 * @param shape 形状
 * @param data 数据
 * @param backwardFunction 反向传播函数
 * @param updateList 需要更新的张量列表
 */
class Tensor<T : NumberTypes>(
    val shape: IntArray,
    val data: Array<T>,
    val backwardFunction: (Tensor<T>) -> Unit = { },
    val updateList: List<Tensor<T>> = emptyList(),
)

/**
 * ## 创建一个标量张量
 *
 * 创建一个一维标量张量
 *
 * @param values 数值
 * @return 张量
 */
inline fun <reified T : NumberTypes> tensorOf(vararg values: T): Tensor<T> {
    return Tensor(
        shape = intArrayOf(values.size),
        data = values.toList().toTypedArray(),
    )
}

fun tensorIntOf(vararg values: Int): Tensor<IntNumber> =
    tensorOf(*values.map { IntNumber(it) }.toTypedArray())

fun tensorDoubleOf(vararg values: Double): Tensor<DoubleNumber> =
    tensorOf(*values.map { DoubleNumber(it) }.toTypedArray())

/**
 * ## 重塑张量形状
 *
 * 重塑张量的形状为指定的维度
 *
 * @param values 新的形状
 * @return 重塑后的张量
 */
fun <T : NumberTypes> Tensor<T>.reshape(vararg values: Int): Tensor<T> =
    Tensor(
        shape = values,
        data = this.data,
    )

/**
 * ## 扩展张量维度
 *
 * 将指定维度转变成一个List返回给调用者
 *
 * 例如，`shape`为`[3,2]`的张量`[1,2,3,4,5,6]`，调用`expand(0)`后，变成了`[[1,2,3],[4,5,6]]`；调用`expand(1)`后，变成了`[[1,4],[2,5],[3,6]]`
 *
 * 注意：该操作不会改变张量的实际数据存储，仅仅是改变了数据的视图
 *
 * @param dimension 维度索引
 * @return `Tensor`列表
 */
inline fun <reified T : NumberTypes> Tensor<T>.expand(dimension: Int): List<Tensor<T>> {
    require(dimension in shape.indices) { "dimension $dimension out of range for shape ${shape.contentToString()}" }
    val size = shape[dimension]
    val resultShape = shape.filterIndexed { idx, _ -> idx != dimension }.toIntArray()
    val result = List(size) { mutableListOf<T>() }
    val strides = IntArray(shape.size) { i ->
        shape.drop(i + 1).fold(1) { acc, v -> acc * v }
    }
    for (idx in data.indices) {
        var remain = idx
        val indices = IntArray(shape.size)
        for (i in shape.indices) {
            indices[i] = remain / strides[i]
            remain %= strides[i]
        }
        result[indices[dimension]].add(data[idx])
    }
    return result.map { Tensor(resultShape, it.toTypedArray()) }
}

fun <T : NumberTypes> Tensor<T>.broadcast(shape: IntArray): Tensor<T> {
    
    TODO()
}