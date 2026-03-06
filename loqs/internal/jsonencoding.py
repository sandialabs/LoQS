#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""JSON tools for serialization.
"""

import json


class JSONEncoderWithErrors(json.JSONEncoder):
    """JSON encoder with more helpful errors.

    This should print out what key/value are causing
    JSON encoding issues.
    """

    def default(self, o):
        try:
            return json.JSONEncoder.default(self, o)
        except TypeError as e:
            print(f"Error serializing object: {o}, Key: {self.key}")
            raise e


def dump_or_dumps_with_error_handling(data, f=None):
    encoder = JSONEncoderWithErrors()
    encoder.key = None  # Initialize key for tracking

    def encode_with_key_tracking(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                encoder.key = key
                encode_with_key_tracking(value)
        else:
            encoder.encode(obj)

    try:
        encode_with_key_tracking(data)
        if f is not None:
            json.dump(data, f, cls=JSONEncoderWithErrors)
        else:
            return json.dumps(data, cls=JSONEncoderWithErrors)
    except TypeError as e:
        print(f"Error during JSON serialization: {e}")
