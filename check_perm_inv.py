import numpy as np


def get_permutation_from_eigenvectors(eigenvectors: np.ndarray) -> np.ndarray:
    """
    Receives N eigenvectors of size N (as an N x N NumPy array) and returns
    a permutation matrix based on a lexicographical tree sort.

    The sorting groups nodes by magnitude, then by sign. Ambiguities between
    signs of the same magnitude are resolved by the max magnitude of their
    respective subtrees (from the next eigenvector).

    Args:
        eigenvectors: An N x N NumPy array where eigenvectors[i, k] is the
                      value of the i-th node in the k-th eigenvector.

    Returns:
        An N x N permutation matrix P, such that P @ A would permute the
        rows of A according to the new sorted order.
    """
    N, num_evs = eigenvectors.shape
    if N != num_evs:
        print(f"Warning: Input is {N}x{num_evs}. "
              f"Expected N x N. Using {num_evs} eigenvectors for {N} nodes.")

    root = {'max_mag': 0.0, 'children': {}}  # This is the "Father" node

    # -----------------------------------------------------------------
    # STEP 1: Build the Tree
    # -----------------------------------------------------------------

    def _build_tree():
        """
        Populates the 'root' dictionary by building the specified tree structure.
        Path: Root -> MagNode -> SignNode -> MagNode -> ... -> Leaf(index)
        """
        for i in range(N):
            current_level_node = root

            for k in range(num_evs):
                value = eigenvectors[i, k]
                mag = np.abs(value)
                sign = np.sign(value)

                # 1. Get or create the MagnitudeNode
                # A "MagnitudeNode" is a dict {'max_mag': 0.0, 'children': {}}
                if mag not in current_level_node['children']:
                    current_level_node['children'][mag] = {
                        'max_mag': 0.0,
                        'children': {}
                    }
                mag_node = current_level_node['children'][mag]

                # 2. Get or create the SignNode
                # A "SignNode" is a dict {'max_mag': 0.0, 'children': {}} or
                # {'max_mag': 0.0, 'leaves': []} if at the last level.
                if sign not in mag_node['children']:
                    if k == num_evs - 1:
                        # Last eigenvector, create a leaf list
                        mag_node['children'][sign] = {'max_mag': 0.0, 'leaves': []}
                    else:
                        # Not the last, create a new children dict for the next level
                        mag_node['children'][sign] = {'max_mag': 0.0, 'children': {}}

                sign_node = mag_node['children'][sign]

                # 3. Add leaf or descend
                if k == num_evs - 1:
                    sign_node['leaves'].append(i)
                else:
                    current_level_node = sign_node  # Descend for next eigenvector

    _build_tree()

    # -----------------------------------------------------------------
    # STEP 2: Sort the Tree (Populate max_magnitude)
    # -----------------------------------------------------------------

    def _calculate_max_mag(node: dict, is_at_sign_node_level: bool) -> float:
        """
        Recursively traverses the tree to populate the 'max_mag' field
        for ambiguity resolution. This is a post-order traversal.

        Args:
            node: The current node (a dict) in the tree.
            is_at_sign_node_level: A boolean to track the level.
                True: 'node' is a SignNode (or root), its children are MagNodes.
                False: 'node' is a MagNode, its children are SignNodes.

        Returns:
            The calculated max_magnitude for this node.
        """
        if 'leaves' in node:
            # Base case: A SignNode at the very end of a branch.
            # It has no subsequent magnitudes, so its max_mag is 0.
            node['max_mag'] = 0.0
            return 0.0

        max_mag_of_subtree = 0.0

        if 'children' in node and node['children']:
            for key, child in node['children'].items():

                if is_at_sign_node_level:
                    # 'node' is a SignNode, 'child' is a MagNode. 'key' is magnitude.
                    # The max_mag for this SignNode is the max magnitude *at this
                    # level* plus the max_mags from all deeper levels.
                    child_subtree_max_mag = _calculate_max_mag(
                        child,
                        is_at_sign_node_level=False
                    )
                    max_mag_of_subtree = max(max_mag_of_subtree, key, child_subtree_max_mag)

                else:
                    # 'node' is a MagNode, 'child' is a SignNode. 'key' is sign.
                    # The max_mag for this MagNode is simply the max of its
                    # children's calculated max_mags.
                    child_subtree_max_mag = _calculate_max_mag(
                        child,
                        is_at_sign_node_level=True
                    )
                    max_mag_of_subtree = max(max_mag_of_subtree, child_subtree_max_mag)

        node['max_mag'] = max_mag_of_subtree
        return max_mag_of_subtree

    # Start the calculation. The root acts like a SignNode
    # (its children are MagNodes).
    _calculate_max_mag(root, is_at_sign_node_level=True)

    # -----------------------------------------------------------------
    # STEP 3: Output (Traverse and Sort)
    # -----------------------------------------------------------------

    def _traverse_and_sort(node: dict, is_at_sign_node_level: bool) -> list[int]:
        """
        Recursively traverses the populated tree and sorts at each
        level to produce the final permutation of indices.

        Args:
            node: The current node (a dict) in the tree.
            is_at_sign_node_level: Boolean to track the level.

        Returns:
            A list of sorted indices from this branch.
        """
        ordered_indices = []

        if 'leaves' in node:
            # Base case: Hit a leaf list.
            # Sort leaves by index for deterministic output if all EV entries
            # were identical.
            return sorted(node['leaves'])

        if 'children' not in node or not node['children']:
            return []

        children_items = list(node['children'].items())

        if is_at_sign_node_level:
            # 'node' is a SignNode (or root). Children are MagNodes (keyed by magnitude).
            # Sort by magnitude (the key), descending.
            sorted_children_items = sorted(
                children_items,
                key=lambda item: item[0],  # item[0] is the magnitude
                reverse=True
            )
        else:
            # 'node' is a MagNode. Children are SignNodes (keyed by sign).
            # This is the ambiguity resolution step.

            def sort_key(item):
                """
                Sorts SignNodes.
                Priority 1: max_mag of the child SignNode (descending).
                Priority 2: sign ('+' > '0' > '-') (descending).
                """
                sign = item[0]
                child_node = item[1]
                max_mag = child_node['max_mag']

                # Give '+' a higher priority than 0, which is
                # higher than '-'.
                sign_priority = {1.0: 2, 0.0: 1, -1.0: 0}

                return (max_mag, sign_priority[sign])

            sorted_children_items = sorted(
                children_items,
                key=sort_key,
                reverse=True  # Sorts by max_mag (desc) then sign_priority (desc)
            )

        # Recurse into the now-sorted children
        for key, child in sorted_children_items:
            ordered_indices.extend(
                _traverse_and_sort(child, is_at_sign_node_level=not is_at_sign_node_level)
            )

        return ordered_indices

    # Start traversal. Root acts like a SignNode.
    permutation_indices = _traverse_and_sort(root, is_at_sign_node_level=True)

    # -----------------------------------------------------------------
    # STEP 4: Create Permutation Matrix
    # -----------------------------------------------------------------

    if len(permutation_indices) != N:
        raise Exception(
            f"Sorting failed: Expected {N} indices, "
            f"but got {len(permutation_indices)}"
        )

    # Create the permutation matrix P
    # P[i, j] = 1 if the i-th row in the new order
    # corresponds to the j-th row in the old order.
    P = np.zeros((N, N), dtype=int)
    for new_row_index, old_row_index in enumerate(permutation_indices):
        P[new_row_index, old_row_index] = 1

    return P


# -----------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------
if __name__ == "__main__":
    print("--- Example 1: Ambiguity Resolution ---")
    # N=4 nodes, 4 eigenvectors
    # Node 0: [ 1,  5, ...]
    # Node 1: [-1,  1, ...]
    # Node 2: [ 1, -8, ...]
    # Node 3: [-1,  9, ...]

    # Tree build:
    # Root
    # -> Mag=1
    #    -> Sign=+1 (Nodes 0, 2)
    #       -> Mag=8 (Node 2)
    #          -> Sign=-1
    #       -> Mag=5 (Node 0)
    #          -> Sign=+1
    #    -> Sign=-1 (Nodes 1, 3)
    #       -> Mag=9 (Node 3)
    #          -> Sign=+1
    #       -> Mag=1 (Node 1)
    #          -> Sign=+1

    # Calculate max_mag (bottom-up):
    # - Sign=+1 (Nodes 0, 2): max_mag = max(8, 5) = 8
    # - Sign=-1 (Nodes 1, 3): max_mag = max(9, 1) = 9

    # Traverse and Sort:
    # 1. At Root: only one child (Mag=1). Descend.
    # 2. At Mag=1: Sort children (Sign=+1, Sign=-1)
    #    - Sign=+1 has max_mag=8
    #    - Sign=-1 has max_mag=9
    #    - Sort key: (max_mag, sign_priority)
    #    - Sign=+1: (8, 2)
    #    - Sign=-1: (9, 0)
    #    - Sorted order (descending): [Sign=-1, Sign=+1]
    # 3. Descend into Sign=-1:
    #    - Children are Mag=9 (Node 3) and Mag=1 (Node 1)
    #    - Sort by magnitude (desc): [Mag=9, Mag=1]
    #    - Traverse Mag=9 -> Sign=+1 -> Leaf [3]
    #    - Traverse Mag=1 -> Sign=+1 -> Leaf [1]
    #    - Result for this branch: [3, 1]
    # 4. Descend into Sign=+1:
    #    - Children are Mag=8 (Node 2) and Mag=5 (Node 0)
    #    - Sort by magnitude (desc): [Mag=8, Mag=5]
    #    - Traverse Mag=8 -> Sign=-1 -> Leaf [2]
    #    - Traverse Mag=5 -> Sign=+1 -> Leaf [0]
    #    - Result for this branch: [2, 0]

    # Final order: [3, 1, 2, 0]
    # This means the new order of nodes is 3, 1, 2, 0.

    evs1 = np.array([
        [1., 5., 0., 0.],  # Node 0
        [-1., 1., 0., 0.],  # Node 1
        [1., -8., 0., 0.],  # Node 2
        [-1., 9., 0., 0.]  # Node 3
    ])

    P1 = get_permutation_from_eigenvectors(evs1)
    print("Eigenvectors 1:\n", evs1)
    print("\nPermutation Matrix P1:\n", P1)
    print("\nPermuted Order (Indices):", P1.argmax(axis=1))

    # Verify permutation
    permuted_evs1 = P1 @ evs1
    print("\nPermuted Eigenvectors (P1 @ evs1):\n", permuted_evs1)
    # The permuted matrix should have rows in order [3, 1, 2, 0]
    # Row 0 (new) = evs1[3] = [-1, 9]
    # Row 1 (new) = evs1[1] = [-1, 1]
    # Row 2 (new) = evs1[2] = [ 1, -8]
    # Row 3 (new) = evs1[0] = [ 1, 5]
    # This looks correct.

    print("\n" + "---" * 10)

    print("\n--- Example 2: Deeper Ambiguity (Same Max Mag) ---")
    # Node 0: [ 1,  5,  10]
    # Node 1: [-1,  5, -20]

    # Tree build:
    # Root
    # -> Mag=1
    #    -> Sign=+1 (Node 0)
    #       -> Mag=5
    #          -> Sign=+1
    #             -> Mag=10
    #                -> Sign=+1 -> Leaf [0]
    #    -> Sign=-1 (Node 1)
    #       -> Mag=5
    #          -> Sign=+1
    #             -> Mag=20
    #                -> Sign=-1 -> Leaf [1]

    # Calculate max_mag (bottom-up):
    # - At EV1, Mag=5, Sign=+1 (for Node 0): child is Mag=10. max_mag=10
    # - At EV1, Mag=5, Sign=+1 (for Node 1): child is Mag=20. max_mag=20
    # - At EV0, Mag=1, Sign=+1 (Node 0): child is Mag=5.
    #   - Its child (Sign=+1) has max_mag=10.
    #   - So Mag=5 node has max_mag=10.
    #   - So Sign=+1 node has max_mag = max(5, 10) = 10.
    # - At EV0, Mag=1, Sign=-1 (Node 1): child is Mag=5.
    #   - Its child (Sign=+1) has max_mag=20.
    #   - So Mag=5 node has max_mag=20.
    #   - So Sign=-1 node has max_mag = max(5, 20) = 20.

    # Traverse and Sort:
    # 1. At Root: only one child (Mag=1). Descend.
    # 2. At Mag=1: Sort children (Sign=+1, Sign=-1)
    #    - Sign=+1 has max_mag=10
    #    - Sign=-1 has max_mag=20
    #    - Sorted order (desc): [Sign=-1, Sign=+1]
    # 3. Descend into Sign=-1 -> Mag=5 -> Sign=+1 -> Mag=20 -> Sign=-1 -> Leaf [1]
    #    - Result: [1]
    # 4. Descend into Sign=+1 -> Mag=5 -> Sign=+1 -> Mag=10 -> Sign=+1 -> Leaf [0]
    #    - Result: [0]

    # Final order: [1, 0]

    evs2 = np.array([
        [1., 5., 10.],  # Node 0
        [-1., 5., -20.],  # Node 1
        [0., 0., 0.]  # Node 2 (to make it 3x3)
    ])

    # Let's analyze [1, 0] vs [0, 0]
    # Node 0: [1, 5, 10]
    # Node 2: [0, 0, 0]
    # Root
    # -> Mag=1 (Node 0)
    # -> Mag=0 (Node 2)
    # Sort by mag (desc): [Mag=1, Mag=0]
    # Final order should start with Node 0, then Node 1, then Node 2.
    # Expected order: [1, 0, 2]

    P2 = get_permutation_from_eigenvectors(evs2)
    print("Eigenvectors 2:\n", evs2)
    print("\nPermutation Matrix P2:\n", P2)
    print("\nPermuted Order (Indices):", P2.argmax(axis=1))

    print("\n" + "---" * 10)
    print("\n--- Example 3: Same Max Mag, Sign Priority ---")
    # Node 0: [ 1,  5,  10]
    # Node 1: [-1,  5, -10]

    # Analysis:
    # At Mag=1, we compare Sign=+1 and Sign=-1
    # - Sign=+1 has max_mag = max(5, 10) = 10
    # - Sign=-1 has max_mag = max(5, 10) = 10
    # Max mags are equal! Use sign priority.
    # '+' (priority 2) > '-' (priority 0)
    # Sorted order: [Sign=+1, Sign=-1]
    # Final order: [0, 1]

    evs3 = np.array([
        [1., 5., 10.],  # Node 0
        [-1., 5., -10.]  # Node 1
    ])

    P3 = get_permutation_from_eigenvectors(evs3)
    print("Eigenvectors 3:\n", evs3)
    print("\nPermutation Matrix P3:\n", P3)
    print("\nPermuted Order (Indices):", P3.argmax(axis=1))