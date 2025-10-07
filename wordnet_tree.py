import argparse
import contextlib
import json
import os
from copy import copy

from nltk.corpus import wordnet as wn


class bcolors:
    """Colors for terminal output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def _lemmas_str(synset):
    return ", ".join([lemma.name() for lemma in synset.lemmas()])


class WNEntry:
    """One wordnet synset."""

    def __init__(
        self,
        name: str,
        id: int,
        lemmas: str,
        parent_id: int,
        depth: int = None,
        in_image_net: bool = False,
        child_ids: list = None,
        in_main_tree: bool = True,
        _n_images: int = 0,
        _description: str = None,
        _name: str = None,
        _pruned: bool = False,
    ):
        self.name = name
        self.id = id
        self.lemmas = lemmas
        self.parent_id = parent_id
        self.depth = depth
        self.in_image_net = in_image_net
        self.child_ids = child_ids
        self.in_main_tree = in_main_tree
        self._n_images = _n_images
        self._description = _description
        self._name = _name
        self._pruned = _pruned

    def __str__(self, tree=None, accumulate=True, colors=True, max_depth=0, max_children=None):
        green = f"{bcolors.OKGREEN}" if colors else ""
        red = f"{bcolors.FAIL}" if colors else ""
        end = f"{bcolors.ENDC}" if colors else ""
        start_symb = f"{green}+{end}" if self.in_image_net else f"{red}-{end}"
        n_ims = f"{self._n_images} of Σ {self.n_images(tree)}" if accumulate and tree is not None else self._n_images
        if self.child_ids is None or tree is None or max_depth == 0:
            return f"{start_symb}{self.name} ({self.id}) > {n_ims}"

        children = self.child_ids
        if max_children is not None and len(children) > max_children:
            children = children[:max_children]
        return f"{start_symb}{self.name} ({self.id}) > {n_ims}\n  " + "\n  ".join(
            [
                "\n  ".join(
                    tree.nodes[child_id]
                    .__str__(tree=tree, accumulate=accumulate, colors=colors, max_depth=max_depth - 1)
                    .split("\n")
                )
                for child_id in children
            ]
        )

    def tree_diff(self, tree_1, tree_2):
        if tree_2[self.id]._n_images > tree_1[self.id]._n_images:
            start_symb = f"{bcolors.OKGREEN}+{bcolors.ENDC}"
        elif tree_2[self.id]._n_images < tree_1[self.id]._n_images:
            start_symb = f"{bcolors.FAIL}-{bcolors.ENDC}"
        else:
            start_symb = f"{bcolors.OKBLUE}={bcolors.ENDC}"
        n_ims = (
            f"{tree_1[self.id]._n_images} + {tree_2[self.id]._n_images - tree_1[self.id]._n_images}  of Σ"
            f" {tree_1[self.id].n_images(tree_2)}/{tree_2[self.id].n_images(tree_2)}"
        )

        if self.child_ids is None:
            return f"{start_symb}{self.name} ({self.id}) > {n_ims}"

        return f"{start_symb}{self.name} ({self.id}) > {n_ims}\n  " + "\n  ".join(
            ["\n  ".join(tree_1.nodes[child_id].tree_diff(tree_1, tree_2).split("\n")) for child_id in self.child_ids]
        )

    def prune(self, tree):
        if self._pruned or self.parent_id is None:
            return

        if self.child_ids is not None:
            for child_id in self.child_ids:
                tree[child_id].prune(tree)

        self._pruned = True
        parent_node = tree.nodes[self.parent_id]
        try:
            parent_node.child_ids.remove(self.id)
        except ValueError as e:
            print(
                f"Error removing {self.name} from"
                f" {parent_node.name} ({[tree[cid].name for cid in parent_node.child_ids]}): {e}"
            )
        while parent_node._pruned:
            parent_node = tree.nodes[parent_node.parent_id]
        parent_node._n_images += self._n_images
        self._n_images = 0

    @property
    def description(self):
        if not self._description:
            self._description = wn.synset_from_pos_and_offset("n", self.id).definition()
        return self._description

    @property
    def print_name(self):
        return self.name.split(".")[0]

    @property
    def is_leaf(self):
        return self.child_ids is None or len(self.child_ids) == 0

    def get_branch(self, tree=None):
        if self.parent_id is None or tree is None:
            return self.print_name

        parent = tree.nodes[self.parent_id]
        return parent.get_branch(tree) + " > " + self.print_name

    def get_branch_list(self, tree):
        if self.parent_id is None:
            return [self]
        parent = tree.nodes[self.parent_id]
        return parent.get_branch_list(tree) + [self]

    def to_dict(self):
        return {
            "name": self.name,
            "id": self.id,
            "lemmas": self.lemmas,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "in_image_net": self.in_image_net,
            "child_ids": self.child_ids,
            "in_main_tree": self.in_main_tree,
            "_n_images": self._n_images,
            "_description": self._description,
            "_name": self._name,
            "_pruned": self._pruned,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def n_images(self, tree=None):
        if tree is None or self.child_ids is None or len(self.child_ids) == 0:
            return self._n_images
        return sum([tree.nodes[child_id].n_images(tree) for child_id in self.child_ids]) + self._n_images

    def n_children(self, tree=None):
        if self.child_ids is None:
            return 0
        if tree is None or len(self.child_ids) == 0:
            return len(self.child_ids)
        return len(self.child_ids) + sum([tree.nodes[child_id].n_children(tree) for child_id in self.child_ids])

    def get_examples(self, tree, n_examples=3):
        if self.child_ids is None or len(self.child_ids) == 0:
            return ""
        child_images = {child_id: tree.nodes[child_id].n_images(tree) for child_id in self.child_ids}
        max_images = max(child_images.values())
        if max_images == 0:
            # go on number of child nodes
            child_images = {child_id: tree.nodes[child_id].n_children(tree) for child_id in self.child_ids}
        # sorted childids by number of images
        top_children = [
            child_id for child_id, n_images in sorted(child_images.items(), key=lambda x: x[1], reverse=True)
        ]
        top_children = top_children[: min(n_examples, len(top_children))]
        return ", ".join(
            [f"{tree.nodes[child_id].print_name} ({tree.nodes[child_id].description})" for child_id in top_children]
        )


class WNTree:
    def __init__(self, root=1740, nodes=None):
        if isinstance(root, int):
            root_id = root
            root_synset = wn.synset_from_pos_and_offset("n", root)
            root_node = WNEntry(
                root_synset.name(),
                root_id,
                _lemmas_str(root_synset),
                parent_id=None,
                depth=0,
            )
        else:
            assert isinstance(root, WNEntry)
            root_id = root.id
            root_node = root

        self.root = root_node
        self.nodes = {root_id: self.root} if nodes is None else nodes
        self.parentless = []
        self.label_index = None
        self.pruned = set()

    def to_dict(self):
        return {
            "root": self.root.to_dict(),
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "parentless": self.parentless,
            "pruned": list(self.pruned),
        }

    def prune(self, min_images):
        pruned_nodes = set()

        # prune all nodes that have fewer than min_images below them
        for node_id, node in self.nodes.items():
            if node.n_images(self) < min_images:
                pruned_nodes.add(node_id)
                node.prune(self)

        # prune all nodes that have fewer than min_images inside them, after all nodes below have been pruned
        node_stack = [self.root]
        node_idx = 0
        while node_idx < len(node_stack):
            node = node_stack[node_idx]
            if node.child_ids is not None:
                for child_id in node.child_ids:
                    child = self.nodes[child_id]
                    node_stack.append(child)
            node_idx += 1

        # now prune the stack from the bottom up
        for node in node_stack[::-1]:
            # only look at images of that class, not of additional children
            if node.n_images() < min_images:
                pruned_nodes.add(node.id)
                node.prune(self)

        self.pruned = pruned_nodes
        return pruned_nodes

    @classmethod
    def from_dict(cls, d):
        tree = cls()
        tree.root = WNEntry.from_dict(d["root"])
        tree.nodes = {int(node_id): WNEntry.from_dict(node) for node_id, node in d["nodes"].items()}
        tree.parentless = d["parentless"]
        if "pruned" in d:
            tree.pruned = set(d["pruned"])
        return tree

    def add_node(self, node_id, in_in=True):
        if node_id in self.nodes:
            self.nodes[node_id].in_image_net = in_in or self.nodes[node_id].in_image_net
            return

        synset = wn.synset_from_pos_and_offset("n", node_id)

        # print(f"adding node {synset.name()} with id {node_id}")

        hypernyms = synset.hypernyms()
        if len(hypernyms) == 0:
            parent_id = None
            self.parentless.append(node_id)
            main_tree = False
            print(f"--------- no hypernyms for {synset.name()} ({synset.offset()}) ------------")
        else:
            parent_id = synset.hypernyms()[0].offset()
            if parent_id not in self.nodes:
                self.add_node(parent_id, in_in=False)
            parent = self.nodes[parent_id]

            if parent.child_ids is None:
                parent.child_ids = []
            parent.child_ids.append(node_id)
            main_tree = parent.in_main_tree

        depth = self.nodes[parent_id].depth + 1 if parent_id is not None else 0
        node = WNEntry(
            synset.name(),
            node_id,
            _lemmas_str(synset),
            parent_id=parent_id,
            in_image_net=in_in,
            depth=depth,
            in_main_tree=main_tree,
        )

        self.nodes[node_id] = node

    def __len__(self):
        return len(self.nodes)

    def image_net_len(self, only_main_tree=False):
        return sum([node.in_image_net for node in self.nodes.values() if node.in_main_tree or not only_main_tree])

    def max_depth(self, only_main_tree=False):
        return max([node.depth for node in self.nodes.values() if node.in_main_tree or not only_main_tree])

    def __str__(self, colors=True):
        return (
            f"WordNet Tree with {len(self)} nodes, {self.image_net_len()} in ImageNet21k;"
            f" {len(self.parentless)} parentless nodes:\n{self.root.__str__(tree=self, colors=colors)}\nParentless:\n"
            + "\n".join([self.nodes[node_id].__str__(tree=self, colors=colors) for node_id in self.parentless])
        )

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            tree_dict = json.load(f)
        return cls.from_dict(tree_dict)

    def subtree(self, node):
        node_id = self[node].id
        if node_id not in self.nodes:
            return None
        node_queue = [self.nodes[node_id]]
        subtree_ids = set()
        while len(node_queue) > 0:
            node = node_queue.pop(0)
            subtree_ids.add(node.id)
            if node.child_ids is not None:
                node_queue += [self.nodes[child_id] for child_id in node.child_ids]
        subtree_nodes = {node_id: copy(self.nodes[node_id]) for node_id in subtree_ids}
        subtree_root = subtree_nodes[node_id]
        subtree_root.parent_id = None
        depth_diff = subtree_root.depth
        for node in subtree_nodes.values():
            node.depth -= depth_diff
        return WNTree(root=subtree_root, nodes=subtree_nodes)

    def _make_label_index(self):
        self.label_index = sorted(
            [node_id for node_id, node in self.nodes.items() if node.n_images(self) > 0 and not node._pruned]
        )

    def get_label(self, node_id):
        if self.label_index is None:
            self._make_label_index()
        while self.nodes[node_id]._pruned:
            node_id = self.nodes[node_id].parent_id
        return self.label_index.index(node_id)

    def n_labels(self):
        if self.label_index is None:
            self._make_label_index()
        return len(self.label_index)

    def __contains__(self, item):
        if isinstance(item, str):
            if item[0] == "n":
                item = int(item[1:])
            else:
                return False
        if isinstance(item, int):
            return item in self.nodes
        if isinstance(item, WNEntry):
            return item.id in self.nodes
        return False

    def __getitem__(self, item):
        if isinstance(item, str) and item[0].startswith("n"):
            with contextlib.suppress(ValueError):
                item = int(item[1:])
        if isinstance(item, str) and ".n." in item:
            for node in self.nodes.values():
                if item == node.name:
                    return node
            raise KeyError(f"Item {item} not found in tree")
        if isinstance(item, int):
            return self.nodes[item]
        if isinstance(item, WNEntry):
            return self.nodes[item.id]
        raise KeyError(f"Item {item} not found in tree")

    def __iter__(self):
        return iter(self.nodes.keys())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create WordNet tree for ImageNet21k")
    parser.add_argument("--rebuild", "-r", action="store_true", help="Rebuild the tree")
    parser.add_argument(
        "--count_21k",
        "-c",
        action="store_true",
        help="Count images in ImageNet21k dataset",
    )
    parser.add_argument("--animals", "-a", action="store_true", help="Make animals subtree")
    args = parser.parse_args()

    load_file = "wordnet_data/imagenet21k_tree_rud.json"
    if os.path.isfile("wordnet_data/imagenet21k_masses_tree.json"):
        load_file = "wordnet_data/imagenet21k_masses_tree.json"

    if os.path.isfile(load_file) and not args.rebuild:
        tree = WNTree.load(load_file)
        print("Tree loaded.")
        print(f"nodes: {len(tree.nodes)}; {sorted(list(tree.nodes.keys()))[:10]}...")
        with open("outfile.ansi", "w") as outfile:
            outfile.write(tree.__str__())

    if args.rebuild:
        wn_id_file = "wordnet_data/imagenet21k_wordnet_ids.txt"
        wn_lemma_file = "wordnet_data/imagenet21k_wordnet_lemmas.txt"

        with open(wn_id_file, "r") as f:
            wn_ids = f.readlines()
        wn_ids = [wn_id.strip() for wn_id in wn_ids]
        with open(wn_lemma_file, "r") as f:
            wn_lemmas = f.readlines()

        tree = WNTree()
        for wn_id in wn_ids:
            tree.add_node(int(wn_id[1:]))

        print(f"nodes: {len(tree.nodes)}; {sorted(list(tree.nodes.keys()))[:10]}...")
        with open("outfile.ansi", "w") as outfile:
            outfile.write(tree.__str__())
        print(tree.image_net_len(only_main_tree=True), "ImageNet nodes in main tree")
        print(tree.max_depth(), "max depth")
        tree.save("wordnet_data/imagenet21k_tree_rud.json")

    if args.count_21k:
        # import tarfile
        # with tarfile.open('/ds/images/imagenet21k/winter21_whole.tar.gz', 'r') as tar:
        #     print('opened tarfile')
        #     for member in tqdm(tar.getmembers()):
        #         if member.isfile():
        #             print("member", member.name)
        #             wn_id = int(member.name.split('/')[-1][1:].split('_')[0])
        #             tree.nodes[wn_id]._n_images += 1

        import torch
        from datadings.reader import MsgpackReader
        from datadings.torch import Compose, CompressedToPIL, Dataset
        from main import collate_single
        from tqdm.auto import tqdm

        reader = MsgpackReader("/ds-sds/images/imagenet21k/train.msgpack")
        dataset = Dataset(reader, transforms={"image": Compose([CompressedToPIL()])})

        dataset = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=10,
            collate_fn=collate_single,
        )

        # counting images
        for data in tqdm(dataset):
            tree.nodes[int(data["key"][1:].split("_")[0])]._n_images += 1

        with open("outfile.ansi", "w") as outfile:
            outfile.write(tree.__str__())

        tree.save("wordnet_data/imagenet21k_masses_tree.json")
        print("Tree saved.")

    if args.animals:
        animals_tree = tree.subtree(15388)
        print(
            f"Animals tree with {animals_tree.root.n_images(animals_tree)} images and {animals_tree.n_labels()} labels"
        )
        with open("animals_outfile.ansi", "w") as outfile:
            outfile.write(animals_tree.__str__())
        animals_tree.save("wordnet_data/imagenet21k_tree_animals.json")
