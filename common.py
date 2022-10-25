def frame_for_stan(frame):
    return {"N": frame.shape[0], **frame.to_dict("list")}

def to_identifier_suffix(species):
    return "_" + species.lower().replace(" ", "_")
