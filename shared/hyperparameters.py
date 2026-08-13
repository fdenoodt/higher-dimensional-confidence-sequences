import argparse
import sys
import re



class Hyperparameters:
    @staticmethod
    def get():
        pass

    @staticmethod
    def apply_sanity_checks(h):
        pass

    @staticmethod
    def _adjust_type(val: any) -> any:
        """
        Transform the string value or list of values to the appropriate type (int, float, bool, dict, or str).
        If val is a list, recursively apply type adjustment to each element.
        :param val: The value to adjust
        :return: The value converted to the appropriate type
        """
        if isinstance(val, list):
            # If the input is a list, apply _adjust_type recursively to each element
            return [Hyperparameters._adjust_type(item) for item in val]

        if val == 'True' or val is True:
            return True
        elif val == 'False' or val is False:
            return False
        elif val == 'None':
            return None

        try:
            # Try converting to a number. ('1.0' -> float, '1' -> int)
            # This is important for --limit_train_batches 1.0, where 1.0 is treated as percentage (100% of data)
            # and 1 is treated as 1 batch

            # If the string contains a '.', treat it as a float
            if '.' in str(val):
                return float(val)
            # Otherwise, try converting to int
            return int(val)
        except (ValueError, TypeError):
            # If it can't be converted to a number, check if it's a list or dict
            if isinstance(val, str):
                if val.startswith('[') and val.endswith(']'):
                    try:
                        # Convert string representation of list to actual list
                        list_val = eval(val)
                        if isinstance(list_val, list):
                            return [Hyperparameters._adjust_type(item) for item in list_val]
                    except (SyntaxError, ValueError):
                        pass
                elif val.startswith('{') and val.endswith('}'):  # e.g. {'crop_size':125}
                    try:
                        # Add quotes to keys if they are missing (due to issues with server-side parsing,
                        # so now also support {crop_size:125})
                        # Add quotes to keys if they are missing
                        val = re.sub(r'(\w+):', r'"\1":', val)

                        # add quotes to values if they are missing
                        val = re.sub(r':(\w+)', r':"\1"', val)

                        dict_val = eval(val)
                        if isinstance(dict_val, dict):
                            return {k: Hyperparameters._adjust_type(v) for k, v in dict_val.items()}
                    except (SyntaxError, ValueError):
                        pass
            # If it can't be converted to a number, list, or dict, return as-is
            return val

    @staticmethod
    def overwrite_args_cli(h: dict[str, any]) -> dict[str, any]:
        # Check if the script is being run in a Jupyter notebook
        if 'ipykernel' not in sys.modules:
            # Parse command-line arguments
            parser = argparse.ArgumentParser()
            for key, value in h.items():
                parser.add_argument(f'--{key}', type=str, default=value)

            args = parser.parse_args()

            # Overwrite the default hyperparameters with the command-line arguments
            for key, value in vars(args).items():
                if key in h:
                    new_val = Hyperparameters._adjust_type(value)
                    h[key] = new_val

        # e.g.: python main.py --method uq_through_redundancy --dataset MNIST --num_views 5 --alpha 0.1 --train True
        return h
