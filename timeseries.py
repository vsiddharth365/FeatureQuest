import glob
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# path = ''  # path of files goes here
# csv_files = glob.glob(path + "/*.csv")
# df_list = (pd.read_csv(file) for file in csv_files)
# # Concatenate all DataFrames
# combined_df = pd.concat(df_list, ignore_index=True)
# tag = combined_df.drop_duplicates(subset=["name"])
# tag_list = list(tag["name"])
#
# metadata_df = pd.read_excel('Metadata.xlsx', sheet_name='Tags_Modeling_2')
# # print(metadata_df)
# combined_df['Date_Time'] = pd.to_datetime(combined_df['Date_Time'])
# # print(df)
# combined_df.sort_values(by=['Date_Time'], inplace=True)


def plot(x1, y1, label1, x_label, y_label):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.print_grid()

    # Add traces
    fig.add_trace(go.Scatter(x=x1,
                             y=y1,
                             name=label1),
                  secondary_y=False)

    # Add figure title
    # fig.update_layout(title_text=title)
    # fig.update_layout(xaxis=dict(dtick='M1',tickformat='%Y-%m-%d'))

    # Set x-axis title
    fig.update_xaxes(title_text="<b>" + x_label + "</b>")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>" + y_label + "</b>", secondary_y=False)
    # fig.update_yaxes(title_text="<b>LP Hyd. Return Pressure </b> BarG", secondary_y=False)
    # fig.update_yaxes(title_text="<b>HP Hyd. Return Pressure</b> BarG", secondary_y=True)

    return fig


def custom_html(pdf, metadata_df, i, tag_list):
    ###Filtering unit and description 
    unit = metadata_df.loc[metadata_df['Tag_Name'] == i, 'Units'].iloc[0]
    desc = metadata_df.loc[metadata_df['Tag_Name'] == i, 'Description'].iloc[0]
    print(unit)
    print(desc)
    list_count = len(tag_list)

    ts = plot(x1=pdf['Date_Time'],
              y1=pdf['Value'],
              label1='actual_value_variation',
              x_label="Time",
              y_label="Tag_Value"
              )

    ts.update_xaxes(rangeslider_visible=True,
                    rangeselector=dict(buttons=list([dict(count=1, label="1 M", step="month", stepmode="backward"),
                                                     dict(count=6, label="6 M", step="month", stepmode="backward"),
                                                     dict(count=1, label="1 Y", step="year", stepmode="backward"),
                                                     dict(count=1, label="YTD", step="year", stepmode="todate"),
                                                     dict(step="all")])))

    fig = px.histogram(pdf["Value"], title="Data Distribution for Tag : " + i + " Unit : " + unit + " Description :  " + desc)
    fig.show()
    with open('Timeplot_data' + '.html', 'a') as f:
        f.write("*" * 250)
        f.write('<br/>' + "Tag " + str(tag_list.index(i) + 1) + " of " + str(list_count))
        f.write('<br/>' + "Summary for  " + i)
        f.write('<br/>' + "Unit of tag : " + unit)
        f.write('<br/>' + "Description of tag : " + desc)
        f.write('<br/>' + "Start Date : " + str(pdf['Date_Time'].min()))
        f.write('<br/>' + "End Date : " + str(pdf['Date_Time'].max()))
        # f.write('<br/>' + "Row count for "+i+' is  '+str(pdf[pdf.columns[1]].count()))
        f.write('<br/>' + "Dataframe Head " + pdf[["Date_Time", "Value"]].head().rename(columns={"Date_Time": "Timestamp", "Value": "Value"}).to_html(index=False))
        f.write('<br/>' + "Statistical Data")
        f.write('<br/>' + pdf.describe().to_html())
        f.write('<br/>' + "Time Series plot for tag : " + i)
        f.write(ts.to_html(full_html=False, include_plotlyjs='cdn', default_width='100%', default_height='525px'))
    with open('Histogram_data' + '.html', 'a') as h:
        h.write(fig.to_html(full_html=False, include_plotlyjs='cdn', default_width='100%', default_height='525px'))


# for i in tag_list:
#     pdf = combined_df[combined_df["name"] == i]
#     custom_html(pdf, metadata_df, i, tag_list)
