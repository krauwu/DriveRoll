import os
import posixpath
import fsspec
import fsspec.implementations.local
import moxing as mox


class DirFileSystem(fsspec.AbstractFileSystem):
    protocol = "dir"

    def __init__(self, path=None, fs=None, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.fs = fsspec.implementations.local.LocalFileSystem() if fs is None else fs

    @property
    def sep(self):
        if self._is_obs_path(self.path):
            return "/"
        return self.fs.sep

    def _is_obs_path(self, path):
        return isinstance(path, str) and path.startswith("obs://")

    def _is_abs_local_path(self, path):
        return isinstance(path, str) and os.path.isabs(path)

    def _join_obs_path(self, base, path):
        if not base:
            return path

        if not path:
            return base

        if self._is_obs_path(path):
            return path

        if self._is_abs_local_path(path):
            return path

        base_clean = base.rstrip("/")
        path_clean = self._strip_protocol(path).lstrip("/")
        return base_clean + "/" + path_clean

    def _join_local_path(self, base, path):
        if not base:
            return path

        if not path:
            return base

        if self._is_obs_path(path):
            return path

        if self._is_abs_local_path(path):
            return path

        return self.fs.sep.join((base, self._strip_protocol(path)))

    def _join(self, path):
        if isinstance(path, str):
            if self._is_obs_path(self.path):
                return self._join_obs_path(self.path, path)
            return self._join_local_path(self.path, path)

        return [self._join(_path) for _path in path]

    def _relpath(self, path):
        if isinstance(path, str):
            if not self.path:
                return path

            if path == self.path:
                return ""

            if self._is_obs_path(self.path):
                prefix = self.path.rstrip("/") + "/"
                if path.startswith(prefix):
                    return path[len(prefix):]
                return path

            prefix = self.path + self.fs.sep
            if path.startswith(prefix):
                return path[len(prefix):]
            return path

        return [self._relpath(_path) for _path in path]

    def _obs_isfile(self, path):
        if not mox.file.exists(path):
            return False
        return not mox.file.is_directory(path)

    def _obs_info(self, path):
        stat = mox.file.stat(path)
        name = path.rstrip("/") if getattr(stat, "is_directory", False) else path
        size = 0 if getattr(stat, "is_directory", False) else getattr(stat, "length", 0)
        info = {
            "name": name,
            "size": size,
            "type": "directory" if getattr(stat, "is_directory", False) else "file",
        }
        if hasattr(stat, "mtime_nsec"):
            info["mtime"] = stat.mtime_nsec
        return info

    def _obs_list_full_paths(self, path, recursive=False):
        entries = mox.file.list_directory(path, recursive=recursive)
        out = []
        base = path.rstrip("/")
        for entry in entries:
            entry_clean = entry.lstrip("/")
            out.append(base + "/" + entry_clean)
        return out

    def cp_file(self, path1, path2, **kwargs):
        raise NotImplementedError("cp_file for OBS is not implemented in this DirFileSystem.")

    def copy(self, path1, path2, *args, **kwargs):
        raise NotImplementedError("copy for OBS is not implemented in this DirFileSystem.")

    def pipe(self, path, value=None, **kwargs):
        full_path = self._join(path)
        if self._is_obs_path(full_path):
            if isinstance(value, bytes):
                with mox.file.read(full_path, "wb") as f:
                    f.write(value)
            else:
                with mox.file.read(full_path, "w") as f:
                    f.write(value)
            return
        return self.fs.pipe(full_path, value=value, **kwargs)

    def pipe_file(self, path, value, **kwargs):
        return self.pipe(path, value=value, **kwargs)

    def cat_file(self, path, *args, **kwargs):
        full_path = self._join(path)
        if self._is_obs_path(full_path):
            return mox.file.read(full_path, binary=True)
        return self.fs.cat_file(full_path, *args, **kwargs)

    def cat(self, path, *args, **kwargs):
        if isinstance(path, list):
            result = {}
            for one_path in path:
                result[one_path] = self.cat_file(one_path, *args, **kwargs)
            return result
        return self.cat_file(path, *args, **kwargs)

    def put_file(self, lpath, rpath, **kwargs):
        full_path = self._join(rpath)
        if self._is_obs_path(full_path):
            with open(lpath, "rb") as src:
                data = src.read()
            with mox.file.read(full_path, "wb") as dst:
                dst.write(data)
            return
        return self.fs.put_file(lpath, full_path, **kwargs)

    def put(self, lpath, rpath, *args, **kwargs):
        return self.put_file(lpath, rpath, **kwargs)

    def get_file(self, rpath, lpath, **kwargs):
        full_path = self._join(rpath)
        if self._is_obs_path(full_path):
            data = mox.file.read(full_path, binary=True)
            with open(lpath, "wb") as f:
                f.write(data)
            return
        return self.fs.get_file(full_path, lpath, **kwargs)

    def get(self, rpath, lpath, *args, **kwargs):
        return self.get_file(rpath, lpath, **kwargs)

    def isfile(self, path):
        full_path = self._join(path)
        if self._is_obs_path(full_path):
            return self._obs_isfile(full_path)
        return self.fs.isfile(full_path)

    def isdir(self, path):
        full_path = self._join(path)
        if self._is_obs_path(full_path):
            return mox.file.exists(full_path) and mox.file.is_directory(full_path)
        return self.fs.isdir(full_path)

    def size(self, path):
        full_path = self._join(path)
        if self._is_obs_path(full_path):
            return mox.file.get_size(full_path)
        return self.fs.size(full_path)

    def exists(self, path):
        full_path = self._join(path)
        if self._is_obs_path(full_path):
            return mox.file.exists(full_path)
        return self.fs.exists(full_path)

    def info(self, path, **kwargs):
        full_path = self._join(path)
        if self._is_obs_path(full_path):
            return self._obs_info(full_path)
        return self.fs.info(full_path, **kwargs)

    def ls(self, path, detail=True, **kwargs):
        full_path = self._join(path)
        if self._is_obs_path(full_path):
            children = self._obs_list_full_paths(full_path, recursive=False)
            if detail:
                out = []
                for child in children:
                    child_info = self._obs_info(child)
                    child_info["name"] = self._relpath(child_info["name"])
                    out.append(child_info)
                return out
            return [self._relpath(child) for child in children]

        ret = self.fs.ls(full_path, detail=detail, **kwargs).copy()
        if detail:
            out = []
            for entry in ret:
                entry = entry.copy()
                entry["name"] = self._relpath(entry["name"])
                out.append(entry)
            return out
        return self._relpath(ret)

    def walk(self, path, *args, **kwargs):
        full_path = self._join(path)
        if self._is_obs_path(full_path):
            recursive_entries = self._obs_list_full_paths(full_path, recursive=True)
            dirs_map = {}
            files_map = {}

            for entry in recursive_entries:
                rel = self._relpath(entry)
                parent = posixpath.dirname(rel)
                name = posixpath.basename(rel)

                if mox.file.is_directory(entry):
                    if parent not in dirs_map:
                        dirs_map[parent] = []
                    dirs_map[parent].append(name)
                    if rel not in dirs_map:
                        dirs_map[rel] = []
                    if rel not in files_map:
                        files_map[rel] = []
                else:
                    if parent not in files_map:
                        files_map[parent] = []
                    files_map[parent].append(name)

            all_roots = set(dirs_map.keys()) | set(files_map.keys())
            if "" not in all_roots:
                all_roots.add("")

            for root in sorted(all_roots):
                dirs = sorted(dirs_map.get(root, []))
                files = sorted(files_map.get(root, []))
                yield root, dirs, files
            return

        for item in self.fs.walk(full_path, *args, **kwargs):
            root, dirs, files = item
            yield self._relpath(root), dirs, files

    def glob(self, path, **kwargs):
        detail = kwargs.get("detail", False)
        full_path = self._join(path)

        if self._is_obs_path(full_path):
            raise NotImplementedError("glob for OBS is not implemented in this DirFileSystem.")

        ret = self.fs.glob(full_path, **kwargs)
        if detail:
            return {self._relpath(one_path): info for one_path, info in ret.items()}
        return self._relpath(ret)

    def du(self, path, *args, **kwargs):
        total = kwargs.get("total", True)
        full_path = self._join(path)

        if self._is_obs_path(full_path):
            if total:
                return mox.file.get_size(full_path, recursive=True)

            recursive_entries = self._obs_list_full_paths(full_path, recursive=True)
            result = {}
            for entry in recursive_entries:
                if not mox.file.is_directory(entry):
                    result[self._relpath(entry)] = mox.file.get_size(entry)
            return result

        ret = self.fs.du(full_path, *args, **kwargs)
        if total:
            return ret
        return {self._relpath(one_path): size for one_path, size in ret.items()}

    def find(self, path, *args, **kwargs):
        detail = kwargs.get("detail", False)
        full_path = self._join(path)

        if self._is_obs_path(full_path):
            recursive_entries = self._obs_list_full_paths(full_path, recursive=True)
            if detail:
                result = {}
                for entry in recursive_entries:
                    result[self._relpath(entry)] = self._obs_info(entry)
                return result
            return [self._relpath(entry) for entry in recursive_entries]

        ret = self.fs.find(full_path, *args, **kwargs)
        if detail:
            return {self._relpath(one_path): info for one_path, info in ret.items()}
        return self._relpath(ret)

    def expand_path(self, path, *args, **kwargs):
        full_path = self._join(path)
        if self._is_obs_path(full_path):
            return self._relpath(full_path)
        return self._relpath(self.fs.expand_path(full_path, *args, **kwargs))

    def mkdir(self, path, *args, **kwargs):
        full_path = self._join(path)
        if self._is_obs_path(full_path):
            mox.file.make_dirs(full_path)
            return
        return self.fs.mkdir(full_path, *args, **kwargs)

    def makedirs(self, path, *args, **kwargs):
        full_path = self._join(path)
        if self._is_obs_path(full_path):
            mox.file.make_dirs(full_path)
            return
        return self.fs.makedirs(full_path, *args, **kwargs)

    def rmdir(self, path):
        full_path = self._join(path)
        if self._is_obs_path(full_path):
            mox.file.remove(full_path, recursive=True)
            return
        return self.fs.rmdir(full_path)

    def mv(self, path1, path2, **kwargs):
        raise NotImplementedError("mv for OBS is not implemented in this DirFileSystem.")

    def touch(self, path, **kwargs):
        full_path = self._join(path)
        if self._is_obs_path(full_path):
            with mox.file.read(full_path, "a"):
                pass
            return
        return self.fs.touch(full_path, **kwargs)

    def created(self, path):
        full_path = self._join(path)
        if self._is_obs_path(full_path):
            stat = mox.file.stat(full_path)
            return getattr(stat, "mtime_nsec", None)
        return self.fs.created(full_path)

    def modified(self, path):
        full_path = self._join(path)
        if self._is_obs_path(full_path):
            stat = mox.file.stat(full_path)
            return getattr(stat, "mtime_nsec", None)
        return self.fs.modified(full_path)

    def sign(self, path, *args, **kwargs):
        full_path = self._join(path)
        if self._is_obs_path(full_path):
            raise NotImplementedError("sign for OBS is not implemented in this DirFileSystem.")
        return self.fs.sign(full_path, *args, **kwargs)

    def __repr__(self):
        return "{}(path='{}', fs={})".format(
            self.__class__.__qualname__, self.path, self.fs)

    def _open(self, path, mode="rb", block_size=None, autocommit=True, cache_options=None, **kwargs):
        full_path = self._join(path)
        if self._is_obs_path(full_path):
            return mox.file.File(full_path, mode)
        return self.fs.open(full_path, mode, **kwargs)