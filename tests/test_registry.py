import pytest

import ls2d
from ls2d.forcing.registry import Registry, SimpleField


def make_registry():
    reg = Registry()
    reg.add_field(SimpleField('a', 'sfc'))
    reg.add_field(SimpleField('b', 'ml'))

    @reg.quantity('x', requires=('sfc:a',), reduce='mean')
    def x(a, ctx):
        return a + 1

    @reg.quantity('y', requires=('x', 'ml:b'), reduce='mean')
    def y(x, b, ctx):
        return x * b

    return reg


def test_resolve_order():
    reg = make_registry()
    order = [getattr(e, 'key', getattr(e, 'name', None)) for e in reg.resolve(['y'])]
    assert order.index('sfc:a') < order.index('x') < order.index('y')
    assert order.index('ml:b') < order.index('y')


def test_required_fields():
    reg = make_registry()
    assert [f.key for f in reg.required_fields(['x'])] == ['sfc:a']
    assert {f.key for f in reg.required_fields(['y'])} == {'sfc:a', 'ml:b'}
    # Already available quantities are not followed.
    assert [f.key for f in reg.required_fields(['y'], available={'x'})] == ['ml:b']


def test_unknown_name_suggests():
    reg = make_registry()
    with pytest.raises(KeyError, match='Did you mean: y'):
        reg.resolve(['yy'])


def test_unknown_requirement():
    reg = make_registry()
    reg.quantity('bad', requires=('nope',))(lambda nope, ctx: nope)
    with pytest.raises(KeyError, match='required by "bad"'):
        reg.resolve(['bad'])


def test_cycle_detection():
    reg = Registry()
    reg.quantity('p', requires=('q',))(lambda q, ctx: q)
    reg.quantity('q', requires=('p',))(lambda p, ctx: p)
    with pytest.raises(ValueError, match='Circular dependency'):
        reg.resolve(['p'])


def test_duplicate_registration():
    reg = make_registry()
    with pytest.raises(KeyError, match='already registered'):
        reg.add_field(SimpleField('a', 'sfc'))
    with pytest.raises(KeyError, match='already registered'):
        reg.quantity('x')(lambda ctx: None)
    reg.quantity('x', replace=True)(lambda ctx: None)


def test_invalid_definitions():
    with pytest.raises(ValueError):
        SimpleField('a', 'xx')
    with pytest.raises(ValueError):
        Registry().quantity('x', stage='column', reduce='mean')(lambda ctx: None)
    with pytest.raises(ValueError):
        Registry().quantity('a:b')(lambda ctx: None)


def test_evaluate_is_pure():
    reg = make_registry()
    inputs = {'sfc:a': 1.0, 'ml:b': 3.0}
    env = reg.evaluate(['y'], inputs, ctx=None)
    assert env['y'] == 6.0
    assert inputs == {'sfc:a': 1.0, 'ml:b': 3.0}


def test_evaluate_missing_era5_field():
    reg = make_registry()
    with pytest.raises(KeyError, match='ml:b'):
        reg.evaluate(['y'], {'sfc:a': 1.0}, ctx=None)


def test_default_registry_subsets():
    # Only what is needed is requested.
    assert {f.key for f in ls2d.era5.registry.required_fields(['ug', 'vg'])} == {'pl:z', 'sfc:sp'}
    assert {f.key for f in ls2d.required_era5_fields(['ps'])} == {'sfc:sp'}
    assert {f.key for f in ls2d.required_era5_fields(['wq'])} == {'sfc:ie', 'sfc:sp', 'sfc:skt', 'ml:q'}
    # All LES outputs need the full catalogue.
    assert len(ls2d.required_era5_fields()) == len(ls2d.era5.registry.fields())


def test_column_names():
    assert ls2d.column_names(['p_lay', 'ps']) == ['p', 'ps']
    assert ls2d.column_names(['wls', 'p_lay']) == ['w', 'z', 'p']
    assert ls2d.column_names(['dtu_total']) == ['dtu_total']
    with pytest.raises(ValueError, match='without `reduce`'):
        ls2d.column_names(['rho'])
    with pytest.raises(KeyError):
        ls2d.column_names(['does_not_exist'])
