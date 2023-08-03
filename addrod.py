def add_item(items_list):
    item = input("What will be added?: ")
    items_list.append(item)


def remove_item(items_list):
    num_items = len(items_list)
    print(f"There are {num_items} items in the list.")
    item_index = input("Which item is deleted?: ")

    try:
        item_index = int(item_index)
        if 0 < item_index <= num_items:
            items_list.pop(item_index - 1)
        else:
            print("Incorrect selection.")
    except ValueError:
        print("Incorrect selection.")


def print_list(items_list):
    print("The following items remain in the list:")
    for item in items_list:
        print(item)


def main():
    items_list = []

    while True:
        print("\n>>\nWould you like to")
        print("(1) Add or")
        print("(2) Remove items or")
        print("(3) Quit?: ", end="")

        choice = input()

        if choice == '1':
            add_item(items_list)
        elif choice == '2':
            remove_item(items_list)
        elif choice == '3':
            print_list(items_list)
            break
        else:
            print("Incorrect selection.")


if __name__ == "__main__":
    main()
