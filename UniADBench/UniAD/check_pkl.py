if __name__ == '__main__':
    import pickle
    from pprint import pprint

    # file_path = "<path-to-local-resource>"
    file_path = "<path-to-local-resource>"
    # file_path = "<path-to-local-resource>"
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    print(type(data))
    print(data.keys())
    # print(data.values())

    metadata = data["anchor"]

    print("\n===== infos type =====")
    print(type(metadata))

    print(metadata)
    print(len(data['frame_mapping']))
    # print("\n===== first 1 infos entries =====")
    # for i, item in enumerate(list(metadata)[:1]):
    #     print(f"\n--- infos[{i}] ---")
    #     keys= list(item.keys())
    #     for j in keys:
    #         print(j)