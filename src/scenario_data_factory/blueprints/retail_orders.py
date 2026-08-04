from __future__ import annotations

from datetime import date

from scenario_data_factory.blueprints.base import BlueprintMetadata, DomainBlueprint
from scenario_data_factory.models.scenario import (
    ColumnSpec,
    ColumnType,
    RelationshipSpec,
    ScenarioSpec,
    TableSpec,
    TimelineSpec,
)


class RetailOrdersBlueprint(DomainBlueprint):
    metadata = BlueprintMetadata(
        domain="retail_orders",
        name="Retail orders",
        description="Customers, products, orders, and order lines for ecommerce demos.",
        tables=["customers", "products", "orders", "order_lines"],
    )

    def build(self, *, name: str, seed: int, scale: str = "demo") -> ScenarioSpec:
        sizes = {"small": (100, 50, 300, 900), "demo": (20_000, 1_000, 75_000, 225_000)}
        customers, products, orders, order_lines = sizes.get(scale, sizes["demo"])
        return ScenarioSpec(
            name=name,
            domain=self.metadata.domain,
            seed=seed,
            timeline=TimelineSpec(start_date=date(2026, 1, 1), batches=30),
            tables=[
                TableSpec(
                    name="customers",
                    row_count=customers,
                    columns=[
                        ColumnSpec(name="customer_id", type=ColumnType.LONG, primary_key=True),
                        ColumnSpec(name="email", type=ColumnType.STRING, faker="email"),
                        ColumnSpec(
                            name="province", type=ColumnType.STRING, values=["ON", "QC", "BC", "AB"]
                        ),
                    ],
                ),
                TableSpec(
                    name="products",
                    row_count=products,
                    columns=[
                        ColumnSpec(name="product_id", type=ColumnType.LONG, primary_key=True),
                        ColumnSpec(name="sku", type=ColumnType.STRING),
                        ColumnSpec(
                            name="category",
                            type=ColumnType.STRING,
                            values=["apparel", "home", "electronics"],
                        ),
                    ],
                ),
                TableSpec(
                    name="orders",
                    row_count=orders,
                    columns=[
                        ColumnSpec(name="order_id", type=ColumnType.LONG, primary_key=True),
                        ColumnSpec(name="customer_id", type=ColumnType.LONG, nullable=False),
                        ColumnSpec(name="order_date", type=ColumnType.DATE),
                    ],
                ),
                TableSpec(
                    name="order_lines",
                    row_count=order_lines,
                    columns=[
                        ColumnSpec(name="order_line_id", type=ColumnType.LONG, primary_key=True),
                        ColumnSpec(name="order_id", type=ColumnType.LONG, nullable=False),
                        ColumnSpec(name="product_id", type=ColumnType.LONG, nullable=False),
                        ColumnSpec(
                            name="quantity", type=ColumnType.INTEGER, min_value=1, max_value=8
                        ),
                    ],
                ),
            ],
            relationships=[
                RelationshipSpec(
                    name="customers_orders",
                    parent_table="customers",
                    parent_column="customer_id",
                    child_table="orders",
                    child_column="customer_id",
                ),
                RelationshipSpec(
                    name="orders_lines",
                    parent_table="orders",
                    parent_column="order_id",
                    child_table="order_lines",
                    child_column="order_id",
                ),
                RelationshipSpec(
                    name="products_lines",
                    parent_table="products",
                    parent_column="product_id",
                    child_table="order_lines",
                    child_column="product_id",
                ),
            ],
        )
