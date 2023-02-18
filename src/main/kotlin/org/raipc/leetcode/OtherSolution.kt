package org.raipc.leetcode

import java.util.ArrayDeque

class OtherSolution {
    // 116. Populating Next Right Pointers in Each Node
    fun connect(root: Node?): Node? {
        if (root == null) return null
        val queue = ArrayDeque<Node>().apply { add(root) }
        while (!queue.isEmpty()) {
            var prev: Node? = null
            repeat(queue.size) { _ ->
                val node = queue.removeFirst()
                node.left?.let { queue.addLast(it) }
                node.right?.let { queue.addLast(it) }
                prev?.next = node
                prev = node
            }
        }
        return root
    }

    fun connectRecursive(root: Node?): Node? = root?.apply {
        left?.next = right
        right?.next = next?.left
        connectRecursive(right)
        connectRecursive(left)
    }

    class Node(var `val`: Int) {
        var left: Node? = null
        var right: Node? = null
        var next: Node? = null
    }
}

