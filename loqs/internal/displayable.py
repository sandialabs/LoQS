"""TODO
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Mapping

from loqs.internal import Serializable


class Displayable(Serializable):
    """Base class for all interactively displayable objects.

    This uses the dict version of objects output by
    `to_serializable()` and a tkinter Treeview to have
    an interactive navigatable window.
    """

    def display(self):
        """Launch an interactive viewer for the object.

        This is a blocking operation until the viewer
        window is closed.
        """
        data = self.to_serialization()

        title = f"{self.__class__.__name__} "
        obj_name = getattr(self, "name", None)
        if obj_name is not None:
            title += f"({obj_name}) "
        title += "Viewer"

        app = DisplayableViewer(data, title)
        app.mainloop()


class DisplayableViewer(tk.Tk):
    """TODO"""

    def __init__(self, data: Mapping, title: str = "LoQS Object Viewer"):
        super().__init__()
        self.title(title)
        self.geometry("400x300")

        self.data = data
        self.create_widgets()

    def create_widgets(self):
        self.tree = ttk.Treeview(self)
        self.tree.pack(expand=True, fill="both")

        # Define columns
        self.tree["columns"] = ("Key", "Value")
        self.tree.column("#0", width=0, stretch=tk.NO)
        self.tree.column("Key", anchor=tk.W, width=120)
        self.tree.column("Value", anchor=tk.W, width=250)

        self.tree.heading("#0", text="", anchor=tk.W)
        self.tree.heading("Key", text="Key/Index", anchor=tk.W)
        self.tree.heading("Value", text="Value", anchor=tk.W)

        # Insert data into the treeview
        self.insert_items("", self.data)

        self.tree.bind("<Double-1>", self.on_item_click)

    def insert_items(self, parent_id, item, depth=1):
        """Recursively insert dictionary/list items into the treeview with indentation."""
        if isinstance(item, (list, tuple)):
            list_dict = {}
            for i, v in enumerate(item):
                if isinstance(v, dict) and "class" in v:
                    # This is probably a serialized object
                    key = f"Index {i}: {v['class'].split('.')[-1]}"

                    # Try to get a nice name
                    name = v.get("name", None)
                    if name is None:
                        name = v.get("log", None)

                    if name is not None:
                        key += f" {name}"
                else:
                    key = f"Index {i}"
                list_dict[key] = v
            item = list_dict

        for key, value in item.items():
            # Create an indentation string based on the depth
            indentation = "      " * depth  # 6 spaces for each depth level
            if isinstance(value, (dict, list, tuple)):
                # If the value is a dictionary, insert it as a parent item
                child_id = self.tree.insert(
                    parent_id, "end", text="", values=(indentation + key, "")
                )
                # Insert nested dictionary items
                self.insert_items(child_id, value, depth + 1)
            else:
                # If the value is not a dictionary or list, insert it as a child item
                self.tree.insert(
                    parent_id,
                    "end",
                    text="",
                    values=(indentation + key, value),
                )

    def on_item_click(self, event):
        selected_item = self.tree.selection()
        if selected_item:
            item_id = selected_item[0]
            # Toggle the expansion of the selected item
            if self.tree.item(item_id, "open"):
                self.tree.item(item_id, open=False)
            else:
                self.tree.item(item_id, open=True)
