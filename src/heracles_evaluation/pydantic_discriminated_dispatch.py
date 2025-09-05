from typing import GenericAlias

from plum.parametric import CovariantMeta
from pydantic import BaseModel, ValidationError
from pydantic.functional_validators import WrapValidator


def resolve_descriminated_union(union_type, data: dict):
    """We need a way to understand which member of a union_type is compatible with `data`.
    We need our resolution to match Pydantic's discriminated union resolution. They
    don't expose a function for this, so we just manually try to instantiate each member
    of the union type until one of them works."""
    for t in union_type.__args__:
        try:
            t(**data)
            return t
        except ValidationError:
            continue
    raise ValueError(
        f"No valid instantiation of union type {union_type} from data: {data}"
    )


def discriminated_union_dispatch(union_field):
    """This decorator enables inferring a generic Class type variable based on
    the resolved discriminated union type of a member variable.  Currently only
    a single type variable is supported, but it would be possible to extend to
    multiple generic type vars."""

    def decorator(K):
        @classmethod
        def inference_function(cls, *args, **kwargs):
            if len(args) > 0:
                raise NotImplementedError(
                    "The discriminated union dispatch functionality has been designed with a Pydantic-based workflow in mind, where all classes are initialized with keyword args only. If you need to initialize this class with positional args, you will need to extend the functiona  lity of discriminated_union_dispatch"
                )

            if type(kwargs[union_field]) is dict:
                resolved_type = resolve_descriminated_union(
                    K.model_fields[union_field].annotation, kwargs[union_field]
                )
            else:
                resolved_type = type(kwargs[union_field])
            return resolved_type

        setattr(K, "__infer_type_parameter__", inference_function)
        return K

    return decorator


def has_plum_generics(K):
    """Pydantic cannot recursively validate classes that have been created with
    the plum `parametric` decorator.  The workaround is that every Pydantic
    model that contains another Pydantic model that is a plum parametric as a
    member variable needs to intercept the validate call for this field and
    manually instantiate the member type."""

    if not issubclass(K, BaseModel):
        raise ValueError(
            "The has_plum_generics decorator is only designed for working with pydantic BaseModel classes"
        )

    for name, field in K.model_fields.items():
        # Fields that are plum parametric classes need special construction
        if issubclass(type(field.annotation), CovariantMeta):
            # Note that we need to force early binding (!!!)
            def validator(v, handler, info, constructor=field.annotation):
                return constructor(**v)

            K.model_fields[name].metadata.append(WrapValidator(validator))
        # Fields that are collections of plum parametric classes need special construction
        # e.g., dict[str, MyParametricType], list[MyParametricType], etc
        # I don't think this will handle collections of unions of parametric types so please don't do that :)
        elif issubclass(type(field.annotation), GenericAlias):
            if issubclass(field.annotation.__origin__, dict):
                key_type, value_type = field.annotation.__args__
                if issubclass(type(value_type), CovariantMeta):

                    def validator(v, handler, info, constructor=value_type):
                        constructed_args = {}
                        for key, val in v.items():
                            print("val: ", val)
                            if type(val) is dict:
                                constructed_args[key] = constructor(**val)
                            else:
                                constructed_args[key] = val
                        return constructed_args
                        # return {key: constructor(**val) for key, val in v.items()}

                    K.model_fields[name].metadata.append(WrapValidator(validator))

            if issubclass(field.annotation.__origin__, list):
                element_type = field.annotation.__args__[0]
                if issubclass(type(element_type), CovariantMeta):

                    def validator(v, handler, info, constructor=value_type):
                        return [constructor(**val) for key, val in v.items()]

                    K.model_fields[name].metadata.append(WrapValidator(validator))

    K.model_rebuild(force=True)
    return K
