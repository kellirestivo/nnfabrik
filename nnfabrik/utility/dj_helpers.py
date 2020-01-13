# helper functions for use with DataJoint tables

import warnings
from datetime import datetime
import hashlib
import datajoint as dj
from git import Repo, cmd

def make_hash(config):
    """"
    takes any non-nested dict to return a 32byte hash
        :args: config -- dictionary. Must not contain objects or nested dicts
        :returns: 32byte hash
    """
    hashed = hashlib.md5()
    for k, v in sorted(config.items()):
        hashed.update(str(v).encode())
    return hashed.hexdigest()


def need_to_commit(repo, repo_name=""):
    changed_files = [item.a_path for item in repo.index.diff(None)]
    has_uncommited = bool(changed_files) or bool(repo.untracked_files)

    err_msg = []
    if has_uncommited:
        err_msg.append("\n{}".format(repo_name))
        if repo.untracked_files:
            for f in repo.untracked_files:
                err_msg.append("Untracked: \t" + f)
        if changed_files:
            for f in changed_files:
                err_msg.append("Changed: \t" + f)

    return "\n".join(err_msg)


def get_origin_url(g):
    for remote in g.remote(verbose=True).split("\n"):
        if remote.find("origin") + 1:
            origin_url = remote.split(" ")[0].split("origin\t")[-1]
            return origin_url
        else:
            warnings.warn("The repo does not have any remote url, named origin, specified.")


def check_repo_commit(repo_path):
    repo = Repo(path=repo_path)
    g = cmd.Git(repo_path)
    origin_url = get_origin_url(g)
    repo_name = origin_url.split("/")[-1].split(".")[0]
    err_msg = need_to_commit(repo, repo_name=repo_name)

    if err_msg:
        return '{}_error_msg'.format(repo_name), err_msg

    else:
        sha1, branch = repo.head.commit.name_rev.split()
        commit_date = datetime.fromtimestamp(repo.head.commit.authored_date).strftime("%A %d. %B %Y %H:%M:%S")
        committer_name = repo.head.commit.committer.name
        committer_email = repo.head.commit.committer.email

        return repo_name, {"sha1": sha1, "branch": branch, "commit_date": commit_date,
                          "committer_name": committer_name, "committer_email": committer_email,
                          "origin_url": origin_url}


def gitlog(repos=()):
    """
    A decorator on computed/imported tables.
    Monitors a list of repositories as pointed out by `repos` containing a list of paths to Git repositories. If any of these repositories 
    contained uncommitted changes, the `populate` is interrupted.
    Otherwise, the state of commits associated with all repositoreis are summarized and stored in the associated entry in the GitLog part table.

    Example:
    
    @schema
    @gitlog(['/path/to/repo1', '/path/to/repo2'])
    class MyComputedTable(dj.Computed):
        ...
    
    """
    def gitlog_wrapper(cls):
        # if repos list is empty, skip the modification alltogether
        if len(repos) == 0:
            return cls
            
        class GitLog(dj.Part):
            definition = """
            ->master
            ---
            info :              longblob
            """

        def check_git(self):
            commits_info = {name: info for name, info in [check_repo_commit(repo) for repo in repos]}
            assert len(commits_info) == len(repos)

            if any(['error_msg' in name for name in commits_info.keys()]):
                err_msgs = ["You have uncommited changes."]
                err_msgs.extend([info for name, info in commits_info.items() if 'error_msg' in name])
                err_msgs.append("\nPlease commit the changes before running populate.\n")
                raise RuntimeError('\n'.join(err_msgs))
                
            return commits_info
        
        cls._base_populate = cls.populate
        cls._base_make = cls.make
        cls.check_git = check_git
        cls.GitLog = GitLog
        cls._commits_info = None
        
        def alt_populate(self, *args, **kwargs):
            # the commits info must be attached to the class
            # as table instance is NOT shared between populate and
            # make calls
            self.__class__._commits_info = self.check_git()
            ret = self._base_populate(*args, **kwargs)
            self.__class__._commits_info = None
            return ret

        def alt_make(self, key):
            ret = self._base_make(key)
            if self._commits_info is not None:
                # if there was some Git commit info
                entry = dict(key, info=self._commits_info)
                self.GitLog().insert1(entry)
            return ret

        cls.populate = alt_populate
        cls.make = alt_make

        return cls
        
    return gitlog_wrapper