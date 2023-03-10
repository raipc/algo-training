package org.raipc.leetcode
import java.util.ArrayDeque

// 173. Binary Search Tree Iterator
class BSTIterator(root: TreeNode?) {
    private val stack = ArrayDeque<TreeNode>().also { addLowest(root, it) }

    private fun addLowest(root: TreeNode?, stack: ArrayDeque<TreeNode>) {
        var node = root
        while (node != null) {
            stack.add(node)
            node = node.left
        }
    }

    fun next(): Int {
        val node = stack.removeLast()
        addLowest(node.right, stack)
        return node.`val`
    }

    fun hasNext(): Boolean = stack.isNotEmpty()

}