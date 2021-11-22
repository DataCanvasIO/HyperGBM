## How to customize storage in HyperGBM?

HyperGBM store intermediate data (model files, cache data, etc.) under the *work_dir*,  which is a subdirectory under your system temporary directory by default.

---

* Customize the *work_dir* location

1. Create directory `conf` under the location where you start `hypergbm` or your python script
2. Create file `storage.py` under the `conf` directory, with the content:
   ```
   c.StorageCfg.root = '/your/full/path/for/work_dir'
   ```
3. Run `hypergbm` command or start your python script as normal

---

* Use s3 compatible storage as HyperGBM *work_dir*
1. Install `s3fs`
2. Create directory `conf` under the location where you start `hypergbm` or your python script
3. Create file `storage.py` under the `conf` directory, with the content:
   ```
   c.StorageCfg.kind = 's3'
   c.StorageCfg.root = '/bucket_name/some_path'
   c.StorageCfg.options = """
   {
           "anon": false,
           "client_kwargs": {
                   "endpoint_url": "your_service_address",
                   "aws_access_key_id": "your_access_key",
                   "aws_secret_access_key": "your_secret_access_key"
           }
   }
   ```

Refer to [s3fs](https://s3fs.readthedocs.io/en/latest/) for more installation and connection information.
